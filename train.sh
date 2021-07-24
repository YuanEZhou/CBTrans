#! /bin/bash 
source /zhouyuanen/anaconda3/bin/activate  base
cd  /zhouyuanen/new/cbtic

#stage1
python  train.py   --noamopt --noamopt_warmup 20000   --seq_per_img 5 --batch_size 10 --beam_size 1 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0  --scheduled_sampling_start 0  --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --max_epochs 15     --checkpoint_path   save/transformer-cb-relu-0.4   --id   transformer-cb-relu-0.4   --caption_model  cbt

# # # stage2
# python  train.py    --seq_per_img 5 --batch_size 10 --beam_size 1 --learning_rate 1e-5 --num_layers 6 --input_encoding_size 512 --rnn_size 2048  --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --self_critical_after 0  --max_epochs    30  --start_from   save/nsc-only-l2r--transformer-cb     --checkpoint_path   save/nsc-only-l2r--transformer-cb   --id  nsc-only-l2r--transformer-cb   --caption_model  cbt


