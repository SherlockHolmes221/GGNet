#!/usr/bin/env bash
set -x
cd src

python -u  main.py hoidet \
--exp_id hoidet_hico_baseline \
--arch hourglass  \
--soft_gaussian \
--hard_negative \
--batch_size 23 \
--master_batch 2 \
--lr 1.5e-4  \
--gpus 0 \
--num_workers 16 \
--val_intervals 100000 \
--image_dir images/train2015 \
--dataset hico \
--root_path '/home/xian/Documents/code/GGNet/Dataset' \
--load_model '/home/xian/Documents/code/GGNet/models/ctdet_coco_hg.pth'


python -u test_hoi.py hoidet \
--exp_id hoidet_hico_baseline \
--arch hourglass \
--model_begin 105  \
--model_end 105 \
--gpus 0 \
--image_dir images/test2015 \
--dataset hico \
--test_with_eval \
--save_predictions \
--root_path '/home/xian/Documents/code/GGNet/Dataset' \




