# Compact Bidirectional Transformer for Image Captioning

## Requirements
- Python 3.8
- Pytorch 1.6
- lmdb
- h5py
- tensorboardX

## Prepare Data
1. Please use **git clone --recurse-submodules** to clone this repository and remember to follow initialization steps in coco-caption/README.md.
2. Download the preprocessd dataset from this [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/ESjYq2E7NlJGuyCaNyCSadEBarCtcRtUMR7Nd0UgTIm3-A?e=Rl0Bu2) and extract it to data/.
3. Please download the converted [VinVL](https://github.com/pzzhang/VinVL/blob/main/DOWNLOAD.md#pre-exacted-image-features) feature from this [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/ETEt4voFu-hAkxxwbwgZEz0BrGiDizFmqMTf3wGdWlv0bQ?e=FcqLhg) and place them under data/mscoco_VinVL/. You can also optionally follow this [instruction](https://github.com/ruotianluo/self-critical.pytorch/blob/master/data/README.md#convert-from-peteanderson80s-original-file) to prepare the fixed or adaptive  bottom-up features extracted by Anderson and place them under data/mscoco/ or data/mscoco_adaptive/.
4. Download part checkpoints from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/ER1w9q3ekqpKmiVPW_yL2pABY2TSyb_PoyBK0xDqEHH_zg?e=7Hfwdk) and extract them to save/.

## Offline Evaluation
To reproduce the results of single CBTIC model on Karpathy test split, just run

```
python  eval.py  --model  save/nsc-transformer-cb-VinVL-feat/model-best.pth   --infos_path  save/nsc-transformer-cb-VinVL-feat/infos_nsc-transformer-cb-VinVL-feat-best.pkl      --beam_size   2   --id  nsc-transformer-cb-VinVL-feat   --split test
```
To reproduce the results of ensemble of CBTIC models on Karpathy test split, just run
```
python eval_ensemble.py   --ids   nsc-transformer-cb-VinVL-feat  nsc-transformer-cb-VinVL-feat-seed1   nsc-transformer-cb-VinVL-feat-seed2  nsc-transformer-cb-VinVL-feat-seed3 --weights  1 1 1 1  --beam_size  2   --split  test
```

## Online Evaluation
Please first run
```
python eval_ensemble.py   --split  test  --language_eval 0  --ids   nsc-transformer-cb-VinVL-feat  nsc-transformer-cb-VinVL-feat-seed1   nsc-transformer-cb-VinVL-feat-seed2  nsc-transformer-cb-VinVL-feat-seed3 --weights  1 1 1 1  --input_json  data/cocotest.json  --input_fc_dir data/mscoco_VinVL/cocobu_test2014/cocobu_fc --input_att_dir  data/mscoco_VinVL/cocobu_test2014/cocobu_att   --input_label_h5    data/cocotalk_bw_label.h5    --language_eval 0        --batch_size  128   --beam_size   2   --id   captions_test2014_cbtic_results 
```
and then follow the [instruction](https://cocodataset.org/#captions-eval) to upload results.
## Training
1.  In the first training stage, such as using VinVL feature,  run 
```
python  train.py   --noamopt --noamopt_warmup 20000   --seq_per_img 5 --batch_size 10 --beam_size 1 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0  --scheduled_sampling_start 0  --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --max_epochs 15     --checkpoint_path   save/transformer-cb-VinVL-feat   --id   transformer-cb-VinVL-feat   --caption_model  cbt     --input_fc_dir   data/mscoco_VinVL/cocobu_fc   --input_att_dir   data/mscoco_VinVL/cocobu_att    --input_box_dir    data/mscoco_VinVL/cocobu_box    
```

2. Then in the second training stage, you need two GPUs with 12G memory each, please copy the above pretrained model first

```
cd save
./copy_model.sh  transformer-cb-VinVL-feat    nsc-transformer-cb-VinVL-feat
cd ..
``` 
and then run
```
python  train.py    --seq_per_img 5 --batch_size 10 --beam_size 1 --learning_rate 1e-5 --num_layers 6 --input_encoding_size 512 --rnn_size 2048  --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --self_critical_after 14  --max_epochs    30  --start_from   save/nsc-transformer-cb-VinVL-feat     --checkpoint_path   save/nsc-transformer-cb-VinVL-feat   --id  nsc-transformer-cb-VinVL-feat   --caption_model  cbt    --input_fc_dir   data/mscoco_VinVL/cocobu_fc   --input_att_dir   data/mscoco_VinVL/cocobu_att    --input_box_dir    data/mscoco_VinVL/cocobu_box 
```

## Note
1. Even if  fixing  all random seed, we find that the results of the two runs are still slightly different when using DataParallel on two GPUs. However, the results can be reproduced exactly when using one GPU.
2. If you are interested in the ablation studies, you can use the `git reflog` to list all commits and use `git reset --hard  commit_id` to change to corresponding commit. 

## Citation

```

```

## Acknowledgements
This repository is built upon [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch). Thanks for the released  code.
