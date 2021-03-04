#!/usr/bin/env bash
set -x
cd src

python -u  main.py hoidet --exp_id hoidet_vcoco_baseline \
--arch hourglass  \
--batch_size 7 \
--master_batch 3  \
--lr 5e-5  \
--gpus 0,1 \
--num_workers 16 \
--val_intervals 100000  \
--image_dir images/train2014 \
--dataset vcoco \
--root_path '/home/xian/Documents/code/GGNet/Dataset' \
--load_model '/home/xian/Documents/code/GGNet/models/ctdet_coco_hg.pth'

#python -u test_hoi.py hoidet \
#--arch hourglass \
#--exp_id hoidet_vcoco_baseline \
#--model_begin 100  \
#--model_end 120 \
#--gpus 4 \
#--image_dir images/val2014 \
#--dataset vcoco \
#--test_with_eval \
#--save_predictions \
#--root_path '/home/xian/Documents/code/GGNet/Dataset'