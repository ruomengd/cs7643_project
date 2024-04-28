#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29519 main.py \
    --learning_rate 1e-5 --num_train_epochs 2 --max_length 1024 --method "vanilla"\
    --base_model "pretrained_model/TinyLlama-1.1B-Chat-v1.0"\
    --wandb_name "TinyLlama-1.1B-Chat-v1.0_0.5unifed_vanilla"\
    --log_dir 'model_finetuned/reward_models_unified_vanilla'


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29519 main.py \
    --learning_rate 1e-5 --num_train_epochs 2 --max_length 1024 --method "margin"\
    --base_model "pretrained_model/TinyLlama-1.1B-Chat-v1.0"\
    --wandb_name "TinyLlama-1.1B-Chat-v1.0_0.5unifed_margin"\
    --log_dir 'model_finetuned/reward_models_unified_margin'


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29519 main.py \
    --learning_rate 1e-5 --num_train_epochs 2 --max_length 1024 --method "label_smooth"\
    --base_model "pretrained_model/TinyLlama-1.1B-Chat-v1.0"\
    --wandb_name "TinyLlama-1.1B-Chat-v1.0_0.5unifed_label_smooth0.1"\
    --log_dir 'model_finetuned/reward_models_unified_label_smooth0.1'


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29519 main.py \
    --learning_rate 1e-5 --num_train_epochs 2 --max_length 1024 --method "contrastive"\
    --base_model "pretrained_model/TinyLlama-1.1B-Chat-v1.0"\
    --wandb_name "TinyLlama-1.1B-Chat-v1.0_0.5unifed_contrastive"\
    --log_dir 'model_finetuned/reward_models_unified_contrastive'

