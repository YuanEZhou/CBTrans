#! /bin/bash 
source /zhouyuanen/anaconda3/bin/activate  base
cd  /zhouyuanen/new/cbtic

# For online evaluation
python eval.py  --input_json  data/cocotest.json  --input_fc_dir data/mscoco_VinVL/cocobu_test2014/cocobu_fc --input_att_dir  data/mscoco_VinVL/cocobu_test2014/cocobu_att   --input_label_h5    data/cocotalk_bw_label.h5    --language_eval 0 --model  save/nsc-transformer-cb-VinVL-feat/model-best.pth   --infos_path  save/nsc-transformer-cb-VinVL-feat/infos_nsc-transformer-cb-VinVL-feat-best.pkl    --batch_size  128   --beam_size   2   --id   captions_test2014_cbtic_results  

