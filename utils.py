from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
import evaluate
import numpy as np
import os
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model
from trl import RewardTrainer
from transformers.utils import PaddingStrategy
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt



def build_dataset(data_path, source_name, tokenizer, max_length, split='train', size=None):
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
    
    ds = load_dataset(data_path, split=split)
    ds = filter_by_source_name(ds, source_name)
        
    if split == 'validation':
        ds = ds.filter(lambda example: example['conv_A_rating'] != example['conv_B_rating'], num_proc=30)

    if size is not None:
        ds = ds.select(range(0, size))

    # ds = ds.select(range(0, len(ds), 2))

    def formatting_func(example):
        kwargs = {"padding": 'max_length', "truncation": True, "max_length": max_length, "return_tensors": "pt"}
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
            "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            "margin": chosen_rating - rejected_rating,
        }

    ds = ds.map(formatting_func, batched=False, num_proc=30)
    ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length, num_proc=30)
    ds.set_format(type="torch")
    return ds


def build_dataset_w_label(data_path, source_name, tokenizer, max_length, split='train', size=None):
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
    
    ds = load_dataset(data_path, split=split)
    ds = filter_by_source_name(ds, source_name)
        
    if split == 'validation':
        ds = ds.filter(lambda example: example['conv_A_rating'] != example['conv_B_rating'], num_proc=30)

    if size is not None:
        ds = ds.select(range(0, size))

    # ds = ds.select(range(0, len(ds), 2))

    def formatting_func(example):
        kwargs = {"padding": 'max_length', "truncation": True, "max_length": max_length, "return_tensors": "pt"}
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
        
        # prompt
        index = prompt_plus_chosen_response.find('<|assistant|>\n')
        prompt = prompt_plus_chosen_response[:index+len('<|assistant|>\n')]
        tokens_prompt = tokenizer.encode_plus(prompt, **kwargs)
     
        len_promt_token = len(tokens_prompt["input_ids"][0]) - torch.sum(tokens_prompt["input_ids"][0] == 32000).item()

        # Add Labels
        chosen_labels = tokens_chosen["input_ids"][0].clone()
        chosen_labels[: len_promt_token] = -100
        
        rejected_labels = tokens_rejected["input_ids"][0].clone()
        rejected_labels[: len_promt_token] = -100
        
        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0], "chosen_labels": chosen_labels,
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0], "rejected_labels": rejected_labels,
            "margin": example['conv_A_rating'] - example['conv_B_rating']
        }

    ds = ds.map(formatting_func, batched=False, num_proc=30)
    ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length, num_proc=30)
    ds.set_format(type="torch")
    return ds


def filter_by_source_name(dataset, source_name):
    """
    Filters the dataset to include only entries where 'source_id' == source_id.

    Parameters:
    - dataset: A Hugging Face dataset.

    Returns:
    - A new dataset filtered based on 'source_id' being source_id.
    """
    filtered_dataset = dataset.filter(lambda example: example['source'] == source_name)
    return filtered_dataset


def filter_by_source_id(dataset, source_id):
    """
    Filters the dataset to include only entries where 'source_id' == source_id.

    Parameters:
    - dataset: A Hugging Face dataset.

    Returns:
    - A new dataset filtered based on 'source_id' being source_id.
    """
    filtered_dataset = dataset.filter(lambda example: example['source_id'] == source_id)
    return filtered_dataset


def find_min_max(dataset):
    """
    Find the minimum and maximum values of the 'rating' field in the dataset.
    """
    min_rating = float('inf')
    max_rating = float('-inf')
    for example in dataset:
        rating = example['rating']
        if rating < min_rating:
            min_rating = rating
        if rating > max_rating:
            max_rating = rating
    return min_rating, max_rating


def apply_min_max_normalization(example, min_rating, max_rating, new_min=-10, new_max=10):
    """
    Apply Min-Max normalization to the 'rating' field of a dataset example.
    """
    normalized_rating = new_min + (example['rating'] - min_rating) * (new_max - new_min) / (max_rating - min_rating)
    example['rating'] = normalized_rating
    return example



def draw_distribution(dataset, save_path):

    # Function to extract 'rating' from each sample. This will be passed to `.map`
    def extract_rating(sample):
        return {'rating': sample['rating']}

    # Use .map to apply the function, setting `remove_columns` to drop all columns except 'rating'
    ratings_dataset = dataset.map(extract_rating, remove_columns=['input_ids', 'attention_mask', 'label'])

    # Now, ratings_dataset['rating'] contains only the ratings
    ratings = ratings_dataset['rating']

    # Analyze the distribution of ratings
    mean_rating = ratings.mean()
    median_rating = ratings.median()

    print(f"Mean Rating: {mean_rating}")
    print(f"Median Rating: {median_rating}")

    # Plot the distribution of ratings
    plt.figure(figsize=(10, 6))
    plt.hist(ratings, bins=np.arange(min(ratings), max(ratings) + 1, 1), edgecolor='k')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.grid(True)
    # plt.show()
    # Save the figure to the current directory
    plt.savefig(save_path)
    

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
    
def save_dataset_in_parquet_splits(dataset, num_splits, directory_path):
    split_size = len(dataset) // num_splits
    for i in range(num_splits):
        start = i * split_size
        end = start + split_size if i < num_splits - 1 else len(dataset)
        split = dataset.select(range(start, end))
        file_path = f"{directory_path}/dataset_split_{i}.parquet"
        split.to_parquet(file_path)