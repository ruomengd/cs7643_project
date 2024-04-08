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
    AutoModel,
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
from utils import *
from trainer import *



def train_vanilla(script_args, training_args, tokenizer_name, model_name, data_path):
    
    # Load the value-head model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = False)
    tokenizer.model_max_length = script_args.max_length
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # ## for llama
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>" 
    DEFAULT_UNK_TOKEN = "<unk>" 
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
    
    # Load Dataset
    train_dataset = build_dataset(data_path, 'berkeley-nest/Nectar', tokenizer, script_args.max_length, split='train') 
    eval_dataset = build_dataset(data_path, 'berkeley-nest/Nectar', tokenizer, script_args.max_length, split='validation')
    #######################################################
    print(len(train_dataset), len(eval_dataset))

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    device = Accelerator().local_process_index 

    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, device_map=device, 
        load_in_8bit=True, 
        # attn_implementation="flash_attention_2",
    )
    model.resize_token_embeddings(len(tokenizer))
    print_trainable_parameters(model)
    model.config.pad_token_id = tokenizer.pad_token_id
    print('Model Loading Succeed.')


    # Set Metric For Validation
    accuracy = evaluate.load('./metrics/accuracy')

    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        predictions = np.argmax(predictions, axis=1)
        labels = np.zeros(predictions.shape)
        return accuracy.compute(predictions=predictions, references=labels)



    # Train the model, woohoo.
    trainer = RewardTrainer_vanilla(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding_vanilla(tokenizer=tokenizer, max_length=script_args.max_length),
        peft_config=peft_config
    )

    print_trainable_parameters(trainer.model)
    print('training')
    trainer.train()




def train_contrastive(script_args, training_args, tokenizer_name, model_name, data_path):
    
    # Load the value-head model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = False)
    tokenizer.model_max_length = script_args.max_length
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # ## for llama
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>" 
    DEFAULT_UNK_TOKEN = "<unk>" 
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
    
    # Load Dataset
    train_dataset = build_dataset(data_path, 'berkeley-nest/Nectar', tokenizer, script_args.max_length, split='train') 
    eval_dataset = build_dataset(data_path, 'berkeley-nest/Nectar', tokenizer, script_args.max_length, split='validation')
    #######################################################
    print(len(train_dataset), len(eval_dataset))

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    device = Accelerator().local_process_index 

    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, device_map=device, 
        load_in_8bit=True, 
        # attn_implementation="flash_attention_2",
    )
    model.resize_token_embeddings(len(tokenizer))
    print_trainable_parameters(model)
    model.config.pad_token_id = tokenizer.pad_token_id
    print('Model Loading Succeed.')


    # Set Metric For Validation
    accuracy = evaluate.load('./metrics/accuracy')

    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        predictions = np.argmax(predictions, axis=1)
        labels = np.zeros(predictions.shape)
        return accuracy.compute(predictions=predictions, references=labels)



    # Train the model, woohoo.
    trainer = RewardTrainer_contrastive(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding_vanilla(tokenizer=tokenizer, max_length=script_args.max_length),
        peft_config=peft_config
    )

    print_trainable_parameters(trainer.model)
    print('training')
    trainer.train()

