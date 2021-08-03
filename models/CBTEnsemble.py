# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

from .CaptionModel import CaptionModel
from .AttModel import pack_wrapper, AttModel
from .CBT import *

class CBTEnsemble(CBT):
    def __init__(self, models, weights=None):
        CaptionModel.__init__(self)
        # super(CBTEnsemble, self).__init__()

        self.models = nn.ModuleList(models)
        self.vocab_size = models[0].vocab_size
        self.seq_length = models[0].seq_length
        self.ss_prob = 0
        weights = weights or [1] * len(self.models)
        self.register_buffer('weights', torch.tensor(weights))

    def init_hidden(self, batch_size):
        return [m.init_hidden(batch_size) for m in self.models]

    def embed(self, it):
        return [m.embed(it) for m in self.models]

    def core(self, *args):
        return zip(*[m.core(*_) for m, _ in zip(self.models, zip(*args))])

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state, tmp_att_masks)
        logprobs = torch.stack([F.softmax(m.logit(output[i]), dim=2) for i,m in enumerate(self.models)], 3).mul(self.weights).div(self.weights.sum()).sum(-1).log()

        return logprobs, state

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask = list(zip(*[m._prepare_feature_forward(att_feats, att_masks) for m in self.models]))
        memory = [m.model.encode(att_feats[i], att_masks[i])  for i,m in enumerate(self.models)]

        return [fc_feats[...,:1]] * len(self.models), [att_feat[...,:1]  for att_feat in att_feats], memory, att_masks


    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        trigrams = [] # will be a list of batch_size dictionaries

        seq = fc_feats.new_zeros((batch_size, 2, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, 2, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0: # input <l2r> and <r2l>
                it = fc_feats.new_zeros((batch_size, 2), dtype=torch.long)
                it[:,0] = self.vocab_size -1 
                it[:,1] = self.vocab_size

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # Mess with trigrams
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:,t-3:t-1]
                for i in range(batch_size): # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = seq[i][t-1]
                    if t == 3: # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:,t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda() # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 2.0 # = 4
                logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs, 2)
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs, temperature))
                it = torch.multinomial(prob_prev.view(-1, prob_prev.size(-1)), 1).view(-1,2, 1)
                sampleLogprobs = logprobs.gather(2, it) # gather the logprobs at sampled positions
                # it = it.view(-1).long() # and flatten indices for downstream processing
                it = it.squeeze(-1)
                sampleLogprobs = sampleLogprobs.squeeze(-1)


            # stop when all finished
            if t == 0:
                unfinished = torch.any(it > 0, dim = 1, keepdim=True)
            else:
                unfinished = unfinished * torch.any(it > 0, dim = 1, keepdim=True)
            it = it * unfinished.type_as(it)
            seq[:,:,t] = it
            seqLogprobs[:,:,t] = sampleLogprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break
        return seq, seqLogprobs


    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        # seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        # seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        seq = torch.LongTensor(batch_size,2,self.seq_length).zero_()
        seqLogprobs = torch.FloatTensor(batch_size,2,self.seq_length)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = [p_fc_feats[i][k:k+1].expand(beam_size, p_fc_feats[i].size(1)) for i,m in enumerate(self.models)]
            tmp_att_feats = [p_att_feats[i][k:k+1].expand(*((beam_size,)+p_att_feats[i].size()[1:])).contiguous()  for i,m in enumerate(self.models)]
            tmp_p_att_feats = [pp_att_feats[i][k:k+1].expand(*((beam_size,)+pp_att_feats[i].size()[1:])).contiguous()    for i,m in enumerate(self.models)]
            tmp_att_masks = [p_att_masks[i][k:k+1].expand(*((beam_size,)+p_att_masks[i].size()[1:])).contiguous() for i,m in enumerate(self.models)] if p_att_masks is not None else None
            for t in range(1):
                if t == 0: # input <l2r> and <r2l>
                    it = fc_feats.new_zeros([beam_size, 2], dtype=torch.long)
                    it[:,0] = self.vocab_size -1 
                    it[:,1] = self.vocab_size

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            seq[k,0,:] = self.done_beams[k][0][0]['seq']
            seq[k,1,:] = self.done_beams[k][1][0]['seq'] 
            seqLogprobs[k,0,:] = self.done_beams[k][0][0]['logps']
            seqLogprobs[k,1,:] = self.done_beams[k][1][0]['logps'] 
        # return the samples and their log likelihoods
        return seq, seqLogprobs


    def beam_search(self, init_state, init_logprobs, *args, **kwargs):

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - diversity_lambda
            return unaug_logprobsf

        # does one step of classical beam search

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam
            ys,ix = torch.sort(logprobsf,1,True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    #compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q,c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    local_unaug_logprob = unaug_logprobsf[q,ix[q,c]]
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_unaug_logprob})
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            new_state = [_.clone() for _ in state]
            # new_state = [[_.clone() for _ in state_] for state_ in state]
            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                # for ii in range(len(new_state)):
                #     for state_ix in range(len(new_state[ii])):
                #     #  copy over state in previous beam q to new beam at vix
                #         new_state[ii][state_ix][:, vix] = state[ii][state_ix][:, v['q']] # dimension one is time step
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
            state = new_state
            return beam_seq,beam_seq_logprobs,beam_logprobs_sum,state,candidates

        # Start diverse_beam_search
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        max_ppl = opt.get('max_ppl', 0)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size # beam per group

        # INITIALIZATIONS
        beam_seq_table = [torch.LongTensor(self.seq_length, bdash, 2).zero_() for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length, bdash, 2).zero_() for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash,2) for _ in range(group_size)]

        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        done_beams_table = [[[],[]] for _ in range(group_size)]
        state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state[0]).chunk(group_size, 2)]
        # state_table = list(zip(*[[list(torch.unbind(_)) for _ in torch.stack(init_state_).chunk(group_size, 2)] for init_state_ in init_state]))
        # state_table = list(zip(*[[list(torch.unbind(_)) for _ in torch.stack(init_state_).chunk(group_size, 2)] for init_state_ in init_state]))
        # pdb.set_trace()
        logprobs_table = list(init_logprobs.chunk(group_size, 0))
        # END INIT

        # Chunk elements in the args
        args = list(args)
        # args = [_.chunk(group_size) if _ is not None else [None]*group_size for _ in args]
        # args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        args = [[_.chunk(group_size) if _ is not None else [None]*group_size for _ in args_] for args_ in args] # arg_name, model_name, group_name
        args = [[[args[j][i][k] for i in range(len(self.models))] for j in range(len(args))] for k in range(group_size)] # group_name, arg_name, model_name

        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size): 
                if t >= divm and t <= self.seq_length + divm - 1:
                    # add diversity
                    logprobsf = logprobs_table[divm].data.float()
                    # suppress previous word
                    if decoding_constraint and t-divm > 0:
                        logprobsf.scatter_(2, beam_seq_table[divm][t-divm-1].unsqueeze(2).cuda(), float('-inf'))
                    # suppress UNK tokens in the decoding, the last two are <l2r> and <r2l>
                    logprobsf[:,:,logprobsf.size(2)-3] = logprobsf[:,:, logprobsf.size(2)-3] - 1000  
                    # diversity is added here
                    # the function directly modifies the logprobsf values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-)
                    unaug_logprobsf = add_diversity(beam_seq_table,logprobsf,t,divm,diversity_lambda,bdash)


                    for i in range(unaug_logprobsf.size(1)):

                        # infer new beams
                        beam_seq_table[divm][:,:,i],\
                        beam_seq_logprobs_table[divm][:,:,i],\
                        beam_logprobs_sum_table[divm][:,i],\
                        [state_table[divm][0][:,:,i,:]],\
                        candidates_divm = beam_step(logprobsf[:,i,:],
                                                    unaug_logprobsf[:,i,:],
                                                    bdash,
                                                    t-divm,
                                                    beam_seq_table[divm][:,:,i],
                                                    beam_seq_logprobs_table[divm][:,:, i],
                                                    beam_logprobs_sum_table[divm][:,i],
                                                    [state_table[divm][0][:,:,i,:]])

                        # if time's up... or if end token is reached then copy beams
                        for vix in range(bdash):
                            if beam_seq_table[divm][t-divm,vix,i] == 0 or t == self.seq_length + divm - 1:
                                final_beam = {
                                    'seq': beam_seq_table[divm][: , vix, i].clone(), 
                                    'logps': beam_seq_logprobs_table[divm][:, vix, i].clone(),
                                    'unaug_p': beam_seq_logprobs_table[divm][:, vix, i].sum().item(),
                                    'p': beam_logprobs_sum_table[divm][vix,i].item()
                                }
                                final_beam['p'] = length_penalty(t-divm+1, final_beam['p'])
                                # if max_ppl:
                                #     final_beam['p'] = final_beam['p'] / (t-divm+1)
                                done_beams_table[divm][i].append(final_beam)
                                # don't continue beams from finished sequences
                                beam_logprobs_sum_table[divm][vix,i] = -1000

                    # move the current group one step forward in time
                    
                    it = beam_seq_table[divm][t-divm]
                    logprobs_table[divm], tmp_state = self.get_logprobs_state(it.cuda(), *(args[divm] + [[state_table[divm] for _ in range(len(self.models))]]))
                    state_table[divm] = tmp_state[0]

        # all beams are sorted by their log-probabilities
        # done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        # done_beams = reduce(lambda a,b:a+b, done_beams_table)
        done_beams = []
        for i in range(group_size):
            for j in range(2):
                done_beams.append(sorted(done_beams_table[i][j], key=lambda x: -x['p'])[:bdash])
        
        return done_beams