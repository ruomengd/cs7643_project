from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
import evaluate
import numpy as np
import os
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets
from transformers.trainer_pt_utils import nested_detach
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    LlamaTokenizer,
    LlamaForSequenceClassification,
)
from trl import RewardTrainer
from transformers.utils import PaddingStrategy
from train import *
import os
import argparse

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=16) 
    per_device_eval_batch_size: Optional[int] = field(default=16)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=5e-6)
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The lr scheduler"},)
    max_length: Optional[int] = field(default=1024) 
    
    base_model: Optional[str] =  field(default="./pretrained_model/TinyLlama-1.1B-Chat-v1.0")
    wandb_name: Optional[str] = field(default="./..",)
    log_dir: Optional[str] = field(default='./..')
    


def parse_args() -> ScriptArguments:
    parser = argparse.ArgumentParser(description="Set parameters for model training.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_hf")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--base_model", type=str, default="pretrained_model/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--wandb_name", type=str, default="Some default value")
    parser.add_argument("--log_dir", type=str, default="./..")
    parser.add_argument("--method", type=str, default="vanilla")

    args = parser.parse_args()
    method = args.method
    delattr(args, 'method')
    return ScriptArguments(**vars(args)), method


script_args, method = parse_args()
print("Training with the following configurations:")
# parser = HfArgumentParser(args)
# script_args = parser.parse_args_into_dataclasses()[0]
print(script_args)

model_name = script_args.base_model
tokenizer_name = model_name
data_path = "./data/Unified-Feedback/all"

model_name_split = model_name.split("/")[-1]
output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}_{script_args.learning_rate}"

training_args = TrainingArguments(
    output_dir=os.path.join(output_name, 'logs'),
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    # weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="epoch",
    save_steps=5000,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=False,
    remove_unused_columns=False,
    label_names=[],
    bf16=True,
    logging_strategy="steps",
    logging_steps=10,
    warmup_ratio=0.03,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    run_name=script_args.wandb_name,
    report_to="wandb",
    ddp_find_unused_parameters=False,
)


if method == 'vanilla':
    train_vanilla(script_args, training_args, tokenizer_name, model_name, data_path)
elif method == 'margin':
    train_margin(script_args, training_args, tokenizer_name, model_name, data_path)
elif method == 'label_smooth':
    train_label_smooth(script_args, training_args, tokenizer_name, model_name, data_path)
elif method == 'contrastive':
    train_contrastive(script_args, training_args, tokenizer_name, model_name, data_path)
else:
    print('Error.')
