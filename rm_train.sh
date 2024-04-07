

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch /home/dingruomeng/robust_rm/reward_train_unified_reg.py


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch /home/dingruomeng/robust_rm/reward_train_contrastive.py 

CUDA_VISIBLE_DEVICES=2,3 accelerate launch /home/dingruomeng/robust_rm/eval_reward_unified.py 

CUDA_VISIBLE_DEVICES=5,6 accelerate launch --main_process_port 29511 /home/dingruomeng/robust_rm/eval_reward_unified_reg_hist.py

CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/dingruomeng/robust_rm/eval_reward_unified_reg_hist.py 


CUDA_VISIBLE_DEVICES=5,6 accelerate launch --main_process_port 29511 /home/dingruomeng/robust_rm/main.py 

CUDA_VISIBLE_DEVICES=1,2,3,4 CUDA_LAUNCH_BLOCKING=1 accelerate launch --main_process_port 29509 /home/dingruomeng/robust_rm/main.py

CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python /home/dingruomeng/robust_rm/main.py

CUDA_VISIBLE_DEVICES=1 python /home/dingruomeng/robust_rm/eval_reward_unified_reg_hist.py

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python /home/dingruomeng/robust_rm/main.py