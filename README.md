# CS 7643 Robust Reward Model

This repository contains the implementation for the CS 7643 Robust Reward Model project. It includes various training and evaluation scripts to work with different machine learning techniques such as vanilla reward models, margin loss, label smoothing, and contrastive learning (SimCSE).

## Getting Started

### Prerequisites

#### Environment Setup
Ensure you have a python==3.10 environment ready, with all dependencies installed. The required packages can be installed using:

```bash
pip install -r requirements.txt
```

#### Pretrained Model
Download the TinyLlama 1B model from the provided [link](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/main) and save it into the `./pretrained_model/` directory.

#### Preparing Dataset
Download the Unified-Feedback dataset from the provided [link](https://huggingface.co/datasets/llm-blender/Unified-Feedback) and save it into the `./data/` directory. Set the data path by updating the `main.py` script with the path to your data directory.

### Training

To train the model, execute the training script from the project root:

```bash
bash rm_train.sh
```

For example:
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29519 main.py \
    --learning_rate 1e-5 --num_train_epochs 2 --max_length 1024 --method "vanilla"\
    --base_model "pretrained_model/TinyLlama-1.1B-Chat-v1.0"\
    --wandb_name "TinyLlama-1.1B-Chat-v1.0_0.5unifed_vanilla"\
    --log_dir 'model_finetuned/reward_models_unified_vanilla'
```


### Evaluation
To evaluate the trained model, run:
```bash
bash rm_eval.sh
```
For example:
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29517 eval_reward_unified.py \
    --max_length 1024\
    --base_model "pretrained_model/TinyLlama-1.1B-Chat-v1.0"\
    --peft_name "model_finetuned/reward_models_Nectar_vanilla/checkpoint-22807"\
    --wandb_name "reward_models_Nectar_vanilla"\
    --log_dir 'eval_nectar'\
    --task "unified"
```


### Modules
- `train.py`: Contains the training functions for the vanilla reward model, including specialized functions for margin loss and label smoothing.
- `trainer.py`: Contains the trainer setups, including those for contrastive learning (SimCSE).
- `utils.py`: Includes necessary data processing functions to prepare and manipulate datasets.

