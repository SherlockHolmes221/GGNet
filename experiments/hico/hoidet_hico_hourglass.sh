#!/usr/bin/env bash
set -x
cd src


python -u  main.py hoidet --exp_id hico_ggnet \
--arch hourglassggnet1 \
--hm_rel_dcn25_i_match \
--dataset hico \
--hard_negative \
--soft_gaussian \
--recover_loss_weight \
--refine_weight 0.1 \
--batch_size 15 \
--master_batch 1 \
--lr 1.5e-4 \
--gpus 0,1,2,3,4,5,6,7 \
--num_workers 16 \
--val_intervals 100000  \
--image_dir images/train2015 \
--root_path '/home/xian/Documents/code/GGNet/Dataset' \
--load_model '/home/xian/Documents/code/GGNet/models/ctdet_coco_hg.pth'


python -u test_hoi.py hoidet \
--exp_id hico_ggnet \
--arch hourglassggnet1  \
--hm_rel_dcn25_i_match \
--model_begin 100  \
--model_end 120 \
--gpus 0 \
--image_dir images/test2015 \
--dataset hico \
--test_refine \
--test_with_eval \
--save_predictions