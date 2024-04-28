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
import torch.nn.functional as F
from trl import RewardTrainer
from transformers.utils import PaddingStrategy
from utils import *

@dataclass
class RewardDataCollatorWithPadding_vanilla:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        margins = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            margins.append(feature['margin'])
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
            "margin": margins,
        }
        return batch


class RewardTrainer_vanilla(RewardTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, alpha=0.1):
        # alpha is the smoothing parameter
   
        rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        # loss = - (1 - alpha) * nn.functional.logsigmoid(rewards_j - rewards_k).mean() - alpha * nn.functional.logsigmoid(rewards_k - rewards_j).mean() 
        # loss = -nn.functional.logsigmoid(rewards_j - rewards_k - torch.tensor(inputs["margin"], device=inputs["margin"][0].device).view(-1,1)).mean()

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


class RewardTrainer_margin(RewardTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, alpha=0.1):
        # alpha is the smoothing parameter
   
        rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        # loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        # loss = - (1 - alpha) * nn.functional.logsigmoid(rewards_j - rewards_k).mean() - alpha * nn.functional.logsigmoid(rewards_k - rewards_j).mean() 
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k - torch.tensor(inputs["margin"], device=inputs["margin"][0].device).view(-1,1)).mean()

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


class RewardTrainer_label_smooth(RewardTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, alpha=0.1):
        # alpha is the smoothing parameter
   
        rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        # loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        loss = - (1 - alpha) * nn.functional.logsigmoid(rewards_j - rewards_k).mean() - alpha * nn.functional.logsigmoid(rewards_k - rewards_j).mean() 
        # loss = -nn.functional.logsigmoid(rewards_j - rewards_k - torch.tensor(inputs["margin"], device=inputs["margin"][0].device).view(-1,1)).mean()

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

class RewardTrainer_contrastive(RewardTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, alpha=0.1):
        # alpha is the smoothing parameter
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True)
        
        # simCSE
        ######################################################################################
        beta = 1 # adjust the impact of the contrastive learning loss
        temp = 1 # temperature
        hidden_states = outputs.hidden_states[-1] # (8, 512, 2048)
        h_s = F.dropout(hidden_states, p=0.05, training=True)
        h_t = F.dropout(hidden_states, p=0.05, training=True)
        # Step 1: Compute sentence embeddings by mean pooling over tokens
        sentence_embeddings_s = h_s.mean(dim=1)  # Shape: [8, 2048]
        sentence_embeddings_t = h_t.mean(dim=1)  # Shape: [8, 2048]
        # Step 2: Compute cosine similarity between all pairs
        similarity_matrix = F.cosine_similarity(sentence_embeddings_s.unsqueeze(1), sentence_embeddings_t.unsqueeze(0), dim=2)
        # Step 3: Create labels indicating which entries in the similarity matrix are positives
        labels = torch.arange(similarity_matrix.size(0)).long().to(inputs["margin"][0].device)
        # Step 4: Compute loss
        probs = F.log_softmax(similarity_matrix / temp, dim=1)  # Convert similarities to probabilities using softmax
        # Compute negative log likelihood loss
        c_loss = F.nll_loss(probs, labels)
        ######################################################################################
        # simCSE End
        
        rewards = outputs[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        # loss = - (1 - alpha) * nn.functional.logsigmoid(rewards_j - rewards_k).mean() - alpha * nn.functional.logsigmoid(rewards_k - rewards_j).mean() 
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        
        loss += beta * 0.1 * c_loss
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

  