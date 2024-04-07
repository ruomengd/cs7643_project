from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
import evaluate
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    LlamaTokenizer,
    DataCollatorWithPadding,
)
from transformers.utils import PaddingStrategy
from utils import *
# os.environ["HF_TOKEN"] = ''


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_eval_batch_size: Optional[int] = field(default=4)
    max_length: Optional[int] = field(default=512) 
    # base_model: Optional[str] =  field(default="/home/yangrui/HuggingFace-Download-Accelerator/hf_hub/models--meta-llama--Llama-2-7b-hf")
    base_model: Optional[str] =  field(default="/home/dingruomeng/robust_rm/model_finetuned/TinyLlama-1.1B-Chat-v1.0")
    # peft_name: Optional[str] =  field(default="None")
    peft_name: Optional[str] =  field(default="/home/dingruomeng/robust_rm/model_finetuned/reward_models_Nectar_vanilla/TinyLlama-1.1B-Chat-v1.0_reward_train_reg_hist_Nectar_TinyLlama-1.1B-Chat-v1.0_vanilla_5e-06/logs/tmp-checkpoint-22807")
    # base_model: Optional[str] =  field(default="/home/yangrui/HuggingFace-Download-Accelerator/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.2")
    wandb_name: Optional[str] = field(default="eval_TinyLlama-1.1B-Chat-v1.0_reward_models_Nectar_vanilla",)
    log_dir: Optional[str] = field(default='./eval_TinyLlama-1.1B-Chat-v1.0_reward_models_Nectar_vanilla')
    task: Optional[str] = field(default='unified')

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_name = script_args.base_model
tokenizer_name = model_name
data_path = "/home/dingruomeng/robust_rm/data/Unified-Feedback/all"
accelerator = Accelerator()
print(data_path)
if accelerator.is_main_process and not os.path.exists(os.path.join(script_args.log_dir, script_args.wandb_name + "_" + script_args.task)):
    os.makedirs(os.path.join(script_args.log_dir, script_args.wandb_name + "_" + script_args.task))



def build_dataset(data_path, tokenizer, split='validation', size=None):
    ds = load_dataset(data_path, split=split)
    ds = filter_by_source_name(ds, 'berkeley-nest/Nectar')
    ds = ds.filter(lambda example: example['conv_A_rating'] != example['conv_B_rating'], num_proc=30)

    if size is not None:
        ds = ds.select(range(0, size))

    source_dict = {'argilla/ultrafeedback-binarized-preferences-cleaned':0,
                    'Anthropic/hh-rlhf': 1,
                    'flan_v2_flan2021': 2,
                    'ultrachat': 3,
                    'evol_instruct': 4,
                    'false_qa': 5,
                    'Dahoas/synthetic-instruct-gptj-pairwise': 6,
                    'flan_v2_cot': 7,
                    'flan_v2_p3': 8,
                    'truthful_qa': 9,
                    'lmsys/chatbot_arena_conversations': 10,
                    'openai/summarize_from_feedback(comparisons)': 11,
                    'sharegpt': 12,
                    'flan_v2_niv2': 13,
                    'berkeley-nest/Nectar': 14,
                    'openai/webgpt_comparisons': 15}
    # ds = ds.select(range(0, len(ds), 2)) ############# eval all

    def formatting_func(example):
        kwargs = {"padding": 'max_length', "truncation": True, "max_length": script_args.max_length, "return_tensors": "pt"}
        example['source_id'] = source_dict[example['source']]
        if example['conv_A_rating'] > example['conv_B_rating']:
            chosen_messages = example['conv_A']
            rejected_messages = example['conv_B']
            chosen_rating = example['conv_A_rating']
            rejected_rating = example['conv_B_rating']
            
        else:
            chosen_messages = example['conv_B']
            rejected_messages = example['conv_A']
            chosen_rating = example['conv_B_rating']
            rejected_rating = example['conv_A_rating']
        
        if 'summarize' in example['source']:
            chosen_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + chosen_messages[0]['content'].strip()
            rejected_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + rejected_messages[0]['content'].strip()
        
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        
        return {
            "input_ids": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0], "chosen_rating": chosen_rating,
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0], "rejected_rating": rejected_rating
        }

    ds = ds.map(formatting_func, batched=False, num_proc=30)
    remove_columns = []
    for name in ds.column_names:
        if 'input_ids' not in name and 'attention' not in name and 'source_id' not in name and 'rating' not in name:
            remove_columns.append(name)
    ds = ds.remove_columns(remove_columns)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= script_args.max_length and len(x["input_ids_rejected"]) <= script_args.max_length, num_proc=30)
    ds.set_format(type="torch")
    return ds



# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
model_name_split = model_name.split("/")[-1]
output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}_{script_args.task}"

# Load the value-head model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = False)
tokenizer.model_max_length = script_args.max_length
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
## for llama
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


eval_dataset = build_dataset(data_path, tokenizer, split='validation')
print('size of test dataset: ', len(eval_dataset))
device = Accelerator().local_process_index 
print(device)
# input()

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=1, device_map=device, 
    # load_in_8bit=True, 
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
if os.path.exists(script_args.peft_name):
    model = PeftModel.from_pretrained(model, script_args.peft_name)
if hasattr(model, 'merge_and_unload'):
    model = model.merge_and_unload()

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
eval_data_loader = DataLoader(eval_dataset, batch_size=script_args.per_device_eval_batch_size, drop_last=True, collate_fn=data_collator)
eval_data_loader = accelerator.prepare(eval_data_loader)

full_chosen_prompts = []
full_rejected_prompts = []
full_rewards_chosen = []
full_rewards_rejected = []
full_source_ids = []
pbar = tqdm(total=len(eval_dataset) // script_args.per_device_eval_batch_size // accelerator.num_processes)
with torch.no_grad():
    for i, batch in enumerate(eval_data_loader):
        reward_chosen_tensors = model(batch["input_ids"].to(model.device), attention_mask=batch["attention_mask_chosen"].to(model.device)).logits.reshape(-1)
        reward_rejected_tensors = model(batch["input_ids_rejected"].to(model.device), attention_mask=batch["attention_mask_rejected"].to(model.device)).logits.reshape(-1)
        full_rewards_chosen.extend(reward_chosen_tensors)
        full_rewards_rejected.extend(reward_rejected_tensors)
        full_chosen_prompts.extend(batch['input_ids'])
        full_rejected_prompts.extend(batch['input_ids_rejected'])
        full_source_ids.extend(batch['source_id'])
        pbar.update(1)

full_chosen_prompts = tokenizer.batch_decode(full_chosen_prompts)
full_rejected_prompts = tokenizer.batch_decode(full_rejected_prompts)
full_rewards_chosen = [x.item() for x in full_rewards_chosen]
full_rewards_rejected = [x.item() for x in full_rewards_rejected]
full_source_ids = [x.item() for x in full_source_ids]

accelerator.wait_for_everyone()
all_chosen_prompts = accelerator.gather_for_metrics(full_chosen_prompts)
all_rejected_prompts = accelerator.gather_for_metrics(full_rejected_prompts)
all_rewards_chosen = accelerator.gather_for_metrics(full_rewards_chosen)
all_rewards_rejected = accelerator.gather_for_metrics(full_rewards_rejected)
all_source_ids = accelerator.gather_for_metrics(full_source_ids)


if accelerator.is_main_process:
    evaluation_result = {
        'chosen_prompts': all_chosen_prompts,
        'rejected_prompts': all_rejected_prompts,
        'chosen_rewards': all_rewards_chosen,
        'rejected_rewards': all_rewards_rejected,
        'source_ids': all_source_ids,
    }
    dataframe = pd.DataFrame(evaluation_result)
    accuracy = (dataframe['chosen_rewards'] > dataframe['rejected_rewards']).mean()
    print('accuracy: ', accuracy)
    dataframe.to_csv(os.path.join(script_args.log_dir, script_args.wandb_name + "_" + script_args.task,'eval_data.csv'))
    with open(os.path.join(script_args.log_dir, script_args.wandb_name + "_" + script_args.task,'accuracy.txt'), 'w+') as f:
        f.write(str(accuracy))
    


