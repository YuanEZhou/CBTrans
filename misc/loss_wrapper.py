import torch,pdb
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag,labels_reverse):
        out = {}
        if not sc_flag:
            if self.opt.cbt:
                labels_combine = torch.cat((labels.unsqueeze(1),labels_reverse.unsqueeze(1)),1)
                output  = self.model(fc_feats, att_feats, labels_combine, att_masks)
                loss = self.crit(output.view(-1,output.size(-2),output.size(-1)), labels_combine[:,:,1:].view(-1,labels_combine.size(-1)-1), masks[:,:,1:].view(-1,masks.size(-1)-1))
            elif self.opt.r2l:
                loss = self.crit(self.model(fc_feats, att_feats, labels_reverse, att_masks), labels_reverse[:,1:], masks[:,1:])
            else:
                loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:])
        else:
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
            if self.opt.cbt:
                gen_result = gen_result.view(-1,gen_result.size(-1))
                sample_logprobs = sample_logprobs.view(-1,sample_logprobs.size(-1))
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(self.model, fc_feats, att_feats, att_masks, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            if self.opt.only_l2r_self_critical:
                sample_logprobs = sample_logprobs.view(-1,2,sample_logprobs.size(-1))[:,0,:]
                gen_result = gen_result.view(-1,2,gen_result.size(-1))[:,0,:]
                reward = reward.view(-1,2,reward.size(-1))[:,0,:]
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out
