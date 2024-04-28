
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29517 eval_reward_unified.py \
    --max_length 1024\
    --base_model "pretrained_model/TinyLlama-1.1B-Chat-v1.0"\
    --peft_name "model_finetuned/reward_models_Nectar_vanilla/checkpoint-22807"\
    --wandb_name "reward_models_Nectar_vanilla"\
    --log_dir 'eval_nectar'\
    --task "unified"

