# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer # * Handles the training loop

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image
import numpy as np

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    
@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'

    # * Add for Contrastive Loss
    contrastive_loss: bool = field(default=False)
    coco_ann_path: Optional[str] = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    config_path: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'alignment_projector', 'alignment_encoder', 'denoiser']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def preprocess_qwen_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for input_id, conversation, target in zip(input_ids, conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)

        rounds_len = len(rounds)
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break

            parts[0] += sep

            if has_image:
                round_ids = tokenizer_image_token(rou, tokenizer)
                instruction_ids = tokenizer_image_token(parts[0], tokenizer)
                equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]

                instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
                round_len = len(round_ids)

            else:
                round_ids = tokenizer(rou).input_ids
                instruction_ids = tokenizer(parts[0]).input_ids
                equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]

                instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
                round_len = len(round_ids)

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            if i == 0:
                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            else:
                target[cur_len + 1: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len - 1:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len - 1}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("qwen_2"):
        return preprocess_qwen_2(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.valid_indices = []
        print(f"[LazySupervisedDataset] Image Path: {data_args.image_folder}")
        if data_args.is_multimodal:
            for i, item in enumerate(list_data_dict):
                if 'image' in item:
                    image_path = os.path.join(data_args.image_folder, item['image'])
                    if os.path.exists(image_path):
                        self.valid_indices.append(i)
                else:
                    self.valid_indices.append(i)
            
            # self.valid_indices = self.valid_indices[:128*5100]
            rank0_print(f"Filtered dataset: {len(self.valid_indices)}/{len(list_data_dict)} valid items")
        else:
            self.valid_indices = list(range(len(list_data_dict)))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

        self.contrastive_loss = getattr(data_args, 'contrastive_loss', True)
        self.coco_ann = None
        self.refcoco_ann = None
        
        if self.contrastive_loss:
            # Conta quanti sample COCO/RefCOCO ci sono
            coco_count = sum(1 for idx in self.valid_indices 
                           if 'image' in self.list_data_dict[idx] and 
                           self.list_data_dict[idx]['image'].startswith('coco'))
            refcoco_count = sum(1 for idx in self.valid_indices 
                              if 'image' in self.list_data_dict[idx] and 
                              'refcoco' in self.list_data_dict[idx]['image'].lower())
            
            rank0_print(f"[Contrastive Loss] Found {coco_count} COCO samples, {refcoco_count} RefCOCO samples")
            
            # Carica COCO annotations (lazy loading)
            if coco_count > 0:
                try:
                    from pycocotools.coco import COCO
                    coco_ann_path = getattr(data_args, 'coco_ann_path', 
                                           '/cluster/project/cvg/data/mscoco/annotations/instances_train2017.json')
                    self.coco_ann = COCO(coco_ann_path)
                    rank0_print(f"[Contrastive Loss] Loaded COCO annotations from {coco_ann_path}")
                except Exception as e:
                    rank0_print(f"[WARNING] Failed to load COCO annotations: {e}")
                    self.contrastive_loss = False

    def __len__(self):
        return len(self.valid_indices)

    @property
    def lengths(self):
        length_list = []
        for idx in self.valid_indices:
            sample = self.list_data_dict[idx]
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for idx in self.valid_indices:
            sample = self.list_data_dict[idx]
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # * Function that builds the sample
        # print(f"[NAVIGATION] INSIDE __getitem__ of LazySupervisedDataset")
        real_idx = self.valid_indices[i]
        sources = self.list_data_dict[real_idx]
        
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        coco_flag = False
        segmentation_masks = None
        if 'image' in sources[0]:
            image_file = self.list_data_dict[real_idx]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            original_image = image.copy()

            # * Most common case of image aspect ratio handling
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                # * More critical scenario, the image will be DISTORTED
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)

            # * Carica Masks
            if self._should_use_contrastive(image_file):
                # print(f"[INFO] Loading COCO masks for {image_file}")
                segmentation_masks = self._load_coco_masks(image_file)

                # * Need to convert segmentation_masks into patch_labels (since the model works with patches)
                #patch_labels = self._build_patch_labels(segmentation_masks) # ^ LongTensor (576,) it's a flattened 24x24 grid for LLaVa (each patch has a value which is the corresponding object in the patch)
                #data_dict['patch_labels'] = patch_labels

                if segmentation_masks:
                    if hasattr(processor, 'crop_size'):
                        img_size = processor.crop_size['height']
                    else:
                        img_size = 336

                    # ← PASSA L'IMMAGINE ORIGINALE per gestire aspect ratio
                    patch_labels = self._build_patch_labels(
                        segmentation_masks,
                        original_image,  # ← CRITICO!
                        image_size=img_size,
                        overlap_threshold=0.1
                    )
                
                # ^ Debug: stampa ogni tanto
                if i % 1000 == 0 and len(segmentation_masks) > 0:
                    pass
                    # print(f"[DEBUG] Sample {i}: Loaded {len(segmentation_masks)} masks for {image_file}")
                    # print(f"[DEBUG] First mask shape: {segmentation_masks[0].shape}")

            ## For COCO
            if self.list_data_dict[real_idx]['image'].startswith('coco'):
                coco_flag=True
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[real_idx]))
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[real_idx]:
            data_dict['image'] = image
            if segmentation_masks:
                data_dict['patch_labels'] = patch_labels
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        data_dict['is_coco'] = coco_flag
        # data_dict['segmentation_masks'] = segmentation_masks
        return data_dict
    
    def _build_patch_labels(self, 
                            segmentation_masks: List[np.ndarray],
                            original_image: Image.Image,
                            image_size: int = 336, 
                            patch_size: int = 14,
                            overlap_threshold: float = 0.1,
                            verbose: bool = False) -> torch.Tensor:
        """
        Converte segmentation masks in SINGLE-LABEL patch-level labels.
        
        Per ogni patch, assegna l'oggetto che occupa la MAGGIOR percentuale di area.
        Se nessun oggetto supera overlap_threshold, la patch è etichettata come background (0).
        
        Args:
            segmentation_masks: Lista di maschere (H_orig, W_orig) in spazio immagine originale
            original_image: PIL Image originale
            image_size: Dimensione target dopo preprocessing (336 per LLaVA)
            patch_size: Dimensione patch ViT (14)
            overlap_threshold: Threshold minimo per considerare un oggetto presente
            
        Returns:
            patch_labels: LongTensor (num_patches,) con valori:
                        0 = background (nessun oggetto)
                        1-N = object_id (1-indexed, dove N = len(segmentation_masks))
        """
        orig_width, orig_height = original_image.size
        num_patches_per_side = image_size // patch_size  # 24
        num_patches = num_patches_per_side ** 2  # 576
        
        # ═══════════════════════════════════════════════════════════
        # DEBUG: Info iniziali
        # ═══════════════════════════════════════════════════════════
        if verbose:
            print(f"\n[PATCH_LABELS] ═══ Processing Image (SINGLE-LABEL) ═══")
            print(f"[PATCH_LABELS] Original image size: {orig_width}x{orig_height}")
            print(f"[PATCH_LABELS] Target size: {image_size}x{image_size}")
            print(f"[PATCH_LABELS] Num masks: {len(segmentation_masks) if segmentation_masks else 0}")
            print(f"[PATCH_LABELS] Aspect ratio mode: {self.data_args.image_aspect_ratio}")
            print(f"[PATCH_LABELS] Overlap threshold: {overlap_threshold}")
        
        # ═══════════════════════════════════════════════════════════
        # Edge Case: Nessuna mask
        # ═══════════════════════════════════════════════════════════
        if not segmentation_masks or len(segmentation_masks) == 0:
            if verbose:
                print(f"[PATCH_LABELS] ⚠️ No masks found → all patches = background (0)")
            return torch.zeros(num_patches, dtype=torch.long)
        
        num_objects = len(segmentation_masks)
        
        # ═══════════════════════════════════════════════════════════
        # ASSERT: Verifica input
        # ═══════════════════════════════════════════════════════════
        assert all(isinstance(mask, np.ndarray) for mask in segmentation_masks), \
            "All masks must be numpy arrays"
        assert all(mask.dtype == bool or mask.dtype == np.uint8 for mask in segmentation_masks), \
            "All masks must be bool or uint8"
        assert all(mask.shape == segmentation_masks[0].shape for mask in segmentation_masks), \
            "All masks must have same shape"
        
        # ═══════════════════════════════════════════════════════════
        # DEBUG: Stampa info sulle masks originali
        # ═══════════════════════════════════════════════════════════
        if verbose:
            print(f"[PATCH_LABELS] Original masks shapes:")
            for i, mask in enumerate(segmentation_masks[:3]):
                mask_h, mask_w = mask.shape
                pixels_on = mask.sum()
                coverage = (pixels_on / (mask_h * mask_w)) * 100
                print(f"[PATCH_LABELS]   Mask {i}: {mask_w}×{mask_h}, "
                    f"{pixels_on} pixels ON ({coverage:.1f}% coverage)")
            if len(segmentation_masks) > 3:
                print(f"[PATCH_LABELS]   ... and {len(segmentation_masks)-3} more masks")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 1: Preprocessing masks (stesso di immagine)
        # ═══════════════════════════════════════════════════════════
        if self.data_args.image_aspect_ratio == 'pad':
            square_size = max(orig_width, orig_height)
            
            if orig_width > orig_height:
                pad_top = (square_size - orig_height) // 2
                pad_left = 0
                if verbose:
                    print(f"[PATCH_LABELS] Padding: VERTICAL → square={square_size}x{square_size}, "
                        f"offset=(top={pad_top}, left={pad_left})")
            else:
                pad_top = 0
                pad_left = (square_size - orig_width) // 2
                if verbose:
                    print(f"[PATCH_LABELS] Padding: HORIZONTAL → square={square_size}x{square_size}, "
                        f"offset=(top={pad_top}, left={pad_left})")
            
            resized_masks = []
            for mask in segmentation_masks:
                # Pad a quadrato
                mask_square = np.zeros((square_size, square_size), dtype=np.uint8)
                mask_square[pad_top:pad_top+orig_height, 
                        pad_left:pad_left+orig_width] = mask.astype(np.uint8)
                
                # Resize
                mask_pil = Image.fromarray(mask_square * 255)
                mask_resized = mask_pil.resize((image_size, image_size), Image.NEAREST)
                resized_masks.append(np.array(mask_resized) > 0)
        else:
            if verbose:
                print(f"[PATCH_LABELS] Resize: DIRECT (no padding) → {image_size}x{image_size}")
            resized_masks = []
            for mask in segmentation_masks:
                mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_resized = mask_pil.resize((image_size, image_size), Image.NEAREST)
                resized_masks.append(np.array(mask_resized) > 0)
        
        # ═══════════════════════════════════════════════════════════
        # ASSERT: Verifica preprocessing
        # ═══════════════════════════════════════════════════════════
        assert len(resized_masks) == num_objects, \
            f"Expected {num_objects} masks, got {len(resized_masks)}"
        assert all(mask.shape == (image_size, image_size) for mask in resized_masks), \
            f"All resized masks must be {image_size}x{image_size}"
        
        # ═══════════════════════════════════════════════════════════
        # DEBUG: Info dopo resize
        # ═══════════════════════════════════════════════════════════
        if verbose:
            print(f"[PATCH_LABELS] Resized masks to {image_size}x{image_size}:")
            for i, mask in enumerate(resized_masks[:3]):
                pixels_on = mask.sum()
                coverage = (pixels_on / (image_size * image_size)) * 100
                print(f"[PATCH_LABELS]   Mask {i}: {pixels_on} pixels ON ({coverage:.1f}% coverage)")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: Assegnazione SINGLE-LABEL per patch
        # ═══════════════════════════════════════════════════════════
        # Inizializza a 0 (background)
        patch_labels = torch.zeros(num_patches, dtype=torch.long)
        
        # Matrice per tracciare overlap percentuali (per debug)
        overlap_matrix = np.zeros((num_patches, num_objects), dtype=np.float32)
        
        patch_area = patch_size * patch_size
        
        for patch_idx in range(num_patches):
            patch_row = patch_idx // num_patches_per_side
            patch_col = patch_idx % num_patches_per_side
            
            y_start = patch_row * patch_size
            y_end = y_start + patch_size
            x_start = patch_col * patch_size
            x_end = x_start + patch_size
            
            # Calcola overlap con TUTTI gli oggetti
            overlaps = []
            for obj_idx, mask in enumerate(resized_masks):
                patch_region = mask[y_start:y_end, x_start:x_end]
                overlap_ratio = patch_region.sum() / patch_area
                overlaps.append(overlap_ratio)
                overlap_matrix[patch_idx, obj_idx] = overlap_ratio
            
            # Trova oggetto con MAGGIOR overlap
            max_overlap = max(overlaps)
            max_obj_idx = overlaps.index(max_overlap)
            
            # Assegna SOLO se supera threshold
            if max_overlap >= overlap_threshold:
                # Label 1-indexed: 0=background, 1=obj_0, 2=obj_1, etc.
                patch_labels[patch_idx] = max_obj_idx + 1
            # else: rimane 0 (background)
        
        # ═══════════════════════════════════════════════════════════
        # ASSERT: Verifica output
        # ═══════════════════════════════════════════════════════════
        assert patch_labels.shape == (num_patches,), \
            f"patch_labels shape mismatch: expected ({num_patches},), got {patch_labels.shape}"
        assert patch_labels.dtype == torch.long, \
            f"patch_labels dtype mismatch: expected torch.long, got {patch_labels.dtype}"
        assert patch_labels.min() >= 0, \
            f"patch_labels contains negative values: min={patch_labels.min()}"
        assert patch_labels.max() <= num_objects, \
            f"patch_labels contains invalid object IDs: max={patch_labels.max()}, num_objects={num_objects}"
        
        # ═══════════════════════════════════════════════════════════
        # DEBUG: Statistiche finali
        # ═══════════════════════════════════════════════════════════
        # Count per label
        label_counts = torch.bincount(patch_labels, minlength=num_objects+1)
        # label_counts[0] = num background patches
        # label_counts[i] = num patches assigned to object (i-1)
        
        num_background = label_counts[0].item()
        num_foreground = (num_patches - num_background)
        
        if verbose:
            print(f"[PATCH_LABELS] ─── Final Statistics (SINGLE-LABEL) ───")
            print(f"[PATCH_LABELS] Patch grid: {num_patches_per_side}x{num_patches_per_side} = {num_patches} patches")
            print(f"[PATCH_LABELS] Background patches: {num_background}/{num_patches} "
                f"({100*num_background/num_patches:.1f}%)")
            print(f"[PATCH_LABELS] Foreground patches: {num_foreground}/{num_patches} "
            f"({100*num_foreground/num_patches:.1f}%)")
        
        # Stampa distribuzione per oggetto
        if verbose:
            print(f"[PATCH_LABELS] Distribution per object:")
            for obj_idx in range(num_objects):
                count = label_counts[obj_idx + 1].item()
                percentage = 100 * count / num_patches
                if count > 0 or obj_idx < 3:  # Stampa sempre primi 3, poi solo non-zero
                    print(f"[PATCH_LABELS]   Object {obj_idx}: {count} patches ({percentage:.1f}%)")
        
        # ═══════════════════════════════════════════════════════════
        # DEBUG: Analisi conflitti (patch con overlap multiplo)
        # ═══════════════════════════════════════════════════════════
        # Trova patch dove c'erano più oggetti sopra threshold
        num_objects_above_threshold = (overlap_matrix >= overlap_threshold).sum(axis=1)
        conflicts = num_objects_above_threshold > 1
        num_conflicts = conflicts.sum()
        
        if num_conflicts > 0 and verbose:
            print(f"[PATCH_LABELS] Conflicts resolved: {num_conflicts} patches had multiple objects")
            print(f"[PATCH_LABELS]   (assigned to object with highest overlap)")
            
            # Esempio di conflitto
            conflict_indices = np.where(conflicts)[0]
            if len(conflict_indices) > 0:
                example_idx = conflict_indices[0]
                example_overlaps = overlap_matrix[example_idx]
                above_threshold = example_overlaps >= overlap_threshold
                print(f"[PATCH_LABELS]   Example: Patch {example_idx} had overlaps: ", end="")
                for obj_idx, overlap in enumerate(example_overlaps):
                    if above_threshold[obj_idx]:
                        assigned = (patch_labels[example_idx].item() == obj_idx + 1)
                        marker = "←CHOSEN" if assigned else ""
                        print(f"Obj{obj_idx}={overlap:.2f}{marker} ", end="")
                print()
        
        # ═══════════════════════════════════════════════════════════
        # WARNINGS
        # ═══════════════════════════════════════════════════════════
        if verbose:
            if num_foreground < num_patches * 0.05:
                print(f"[PATCH_LABELS] ⚠️ WARNING: Very few foreground patches (<5%)!")
                print(f"[PATCH_LABELS]   This might indicate misalignment or very small objects")
            
            if num_foreground == 0:
                print(f"[PATCH_LABELS] ❌ ERROR: NO foreground patches assigned!")
                print(f"[PATCH_LABELS]   All objects below threshold or masks misaligned")
            
            # Verifica balance tra oggetti
            if num_objects > 1:
                obj_counts = label_counts[1:].float()  # escludi background
                if obj_counts.max() > 0:
                    imbalance_ratio = obj_counts.max() / obj_counts[obj_counts > 0].min()
                    if imbalance_ratio > 10:
                        print(f"[PATCH_LABELS] ⚠️ WARNING: High object imbalance (ratio={imbalance_ratio:.1f})")
                        print(f"[PATCH_LABELS]   Some objects may be very small or occluded")
        
        print(f"[PATCH] {orig_width}x{orig_height} | {num_objects} objs | "
            f"{num_foreground}/{num_patches} patches ({100*num_foreground/num_patches:.0f}%)")
        if verbose:
            print(f"[PATCH_LABELS] ═══════════════════════════\n")
        
        return patch_labels

    def _load_coco_masks(self, image_filename: str) -> List[np.ndarray]:
        """
        Carica maschere di segmentazione COCO per un'immagine
        
        Args:
            image_filename: es. "coco/train2017/000000123456.jpg"
            
        Returns:
            Lista di maschere numpy (H, W) bool, una per oggetto
        """
        if self.coco_ann is None:
            return []
        
        try:
            # Estrai image_id dal filename
            # Format: "coco/train2017/000000123456.jpg" -> 123456
            image_id_str = image_filename.split('/')[-1].replace('.jpg', '')
            image_id = int(image_id_str)
            
            # Ottieni annotations per questa immagine
            ann_ids = self.coco_ann.getAnnIds(imgIds=[image_id])
            anns = self.coco_ann.loadAnns(ann_ids)
            
            # Converti in maschere
            masks = []
            for ann in anns:
                mask = self.coco_ann.annToMask(ann)
                
                # Filtra oggetti troppo piccoli (< 50 pixel)
                if mask.sum() >= 50:
                    masks.append(mask.astype(bool))
            
            return masks
            
        except Exception as e:
            # Se fallisce (es. image_id non trovato), ritorna lista vuota
            # print(f"[WARNING] Failed to load masks for {image_filename}: {e}")
            return []
    
    def _should_use_contrastive(self, image_filename: str) -> bool:
        """
        Determina se applicare contrastive loss per questa immagine
        """
        if not self.contrastive_loss:
            return False
        
        # COCO images
        if image_filename.startswith('coco'):
            return True
        
        # RefCOCO images (opzionale per futuro)
        # if 'refcoco' in image_filename.lower():
        #     return True
        
        return False

# class LazySupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, data_path: str,
#                  tokenizer: transformers.PreTrainedTokenizer,
#                  data_args: DataArguments):
#         super(LazySupervisedDataset, self).__init__()
#         list_data_dict = json.load(open(data_path, "r"))

#         rank0_print("Formatting inputs...Skip in lazy mode")
#         self.tokenizer = tokenizer
#         self.list_data_dict = list_data_dict
#         self.data_args = data_args

#     def __len__(self):
#         return len(self.list_data_dict)

#     @property
#     def lengths(self):
#         length_list = []
#         for sample in self.list_data_dict:
#             img_tokens = 128 if 'image' in sample else 0
#             length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
#         return length_list

#     @property
#     def modality_lengths(self):
#         length_list = []
#         for sample in self.list_data_dict:
#             cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
#             cur_len = cur_len if 'image' in sample else -cur_len
#             length_list.append(cur_len)
#         return length_list

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         sources = self.list_data_dict[i]
#         if isinstance(i, int):
#             sources = [sources]
#         assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
#         if 'image' in sources[0]:
#             image_file = self.list_data_dict[i]['image']
#             image_folder = self.data_args.image_folder
#             processor = self.data_args.image_processor
#             image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
#             if self.data_args.image_aspect_ratio == 'pad':
#                 def expand2square(pil_img, background_color):
#                     width, height = pil_img.size
#                     if width == height:
#                         return pil_img
#                     elif width > height:
#                         result = Image.new(pil_img.mode, (width, width), background_color)
#                         result.paste(pil_img, (0, (width - height) // 2))
#                         return result
#                     else:
#                         result = Image.new(pil_img.mode, (height, height), background_color)
#                         result.paste(pil_img, ((height - width) // 2, 0))
#                         return result
#                 image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
#                 image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
#             else:
#                 image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
#             sources = preprocess_multimodal(
#                 copy.deepcopy([e["conversations"] for e in sources]),
#                 self.data_args)
#         else:
#             sources = copy.deepcopy([e["conversations"] for e in sources])
#         data_dict = preprocess(
#             sources,
#             self.tokenizer,
#             has_image=('image' in self.list_data_dict[i]))
#         if isinstance(i, int):
#             data_dict = dict(input_ids=data_dict["input_ids"][0],
#                              labels=data_dict["labels"][0])

#         # image exist in the data
#         if 'image' in self.list_data_dict[i]:
#             data_dict['image'] = image
#         elif self.data_args.is_multimodal:
#             # image does not exist in the data, but the model is multimodal
#             crop_size = self.data_args.image_processor.crop_size
#             data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
#         return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        # * Create the batch data structure as dictionary
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # * Add optional images
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images        
        
        is_coco = [instance.get('is_coco', False) for instance in instances]
        batch['is_coco'] = torch.tensor(is_coco, dtype=torch.bool)

        # * Add for contrastive loss
        # * Gestione patch_labels con batching. Devo fare uno stack intelligente
        # ═══════════════════════════════════════════════════════════
        # GESTIONE PATCH_LABELS: alcuni items possono non averle
        # ═══════════════════════════════════════════════════════════
        if any('patch_labels' in inst for inst in instances):
            patch_labels_list = []
            
            # Prima pass: trova il numero massimo di oggetti
            for instance in instances:
                if 'patch_labels' in instance:
                    pl = instance['patch_labels']  # (576,) - single label per patch
            
                    # Validazione
                    assert pl.dim() == 1, f"Expected 1D patch_labels, got shape {pl.shape}"
                    assert pl.dtype == torch.long, f"Expected torch.long, got {pl.dtype}"
                    
                    patch_labels_list.append(pl)
                else:
                    # Nessuna label per questa istanza -> padding con -1
                    patch_labels_list.append(torch.full((576,), -1, dtype=torch.long))
            
            batch['patch_labels'] = torch.stack(patch_labels_list)
            # print(f"[COLLATOR] Batched patch_labels: {batch['patch_labels'].shape} (single-label format)")
        
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    # * Parsing of the arguments (model, data and training)
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    # * Load the multimodal model (LLaVa)
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            # breakpoint()
            if training_args.config_path is not None:
                config = transformers.AutoConfig.from_pretrained(training_args.config_path, trust_remote_code=True)
            else:
                config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

            if 'Qwen2' in model_args.model_name_or_path:
                model = LlavaQwen2ForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
            elif 'vicuna' in model_args.model_name_or_path:
                # * This should be my case
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # * LoRA settings
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # * Handle tokenizer setup
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif "llama-3" in model_args.model_name_or_path.lower():
            tokenizer.pad_token = '<|end_of_text|>'
        else:  # use qwen
            tokenizer.legacy = False
            if tokenizer.pad_token is None:
                print(f"Adding pad token as '<|pad|>'")
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="<|pad|>"),
                    tokenizer=tokenizer,
                    model=model,
                )
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # * Vision Tower Initialization
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # ^ ========================================
    # ^ SINCRONIZZA CONFIG CON DATA_ARGS PRIMA DI MANDARLI A DATALOADER
    # ^ ========================================
    # Se il config ha contrastive_loss abilitato, sincronizza con data_args
    if hasattr(config, 'contrastive_loss') and config.contrastive_loss:
        # Abilita nel dataset
        data_args.contrastive_loss = True
        
        # Assicurati che il path annotations sia settato
        if not hasattr(data_args, 'coco_ann_path') or data_args.coco_ann_path is None:
            data_args.coco_ann_path = "/cluster/project/cvg/data/mscoco/annotations/instances_train2017.json"
        
        # Log per debug
        rank0_print("="*60)
        rank0_print("[CONTRASTIVE LOSS] Configuration from config.json:")
        rank0_print(f"  Enabled: {config.contrastive_loss}")
        rank0_print(f"  Weight: {config.contrastive_weight}")
        rank0_print(f"  Temperature: {config.contrastive_temperature}")
        rank0_print(f"  N Samples: {config.contrastive_n_samples}")
        rank0_print(f"  Sum in Log: {config.contrastive_sum_in_log}")
        rank0_print(f"  COCO Annotations: {data_args.coco_ann_path}")
        rank0_print("="*60)
    else:
        # Disabilita contrastive loss
        data_args.contrastive_loss = False
        rank0_print("[INFO] Contrastive Loss: DISABLED (not in config or set to false)")
    # ^ ========================================

    # * Create the Dataset & DataCollator
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    # * Create the Trainer
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    # * Start the training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        # breakpoint()
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    # if training_args.lora_enable:
    #     state_dict = get_peft_state_maybe_zero_3(
    #         model.named_parameters(), training_args.lora_bias
    #     )
    #     non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
    #         model.named_parameters()
    #     )
    #     if training_args.local_rank == 0 or training_args.local_rank == -1:
    #         model.config.save_pretrained(training_args.output_dir)
    #         model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    #         torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    # * Save modules if LoRA is enabled
    if training_args.lora_enable:
        # breakpoint()
        # 1. 기존 저장 방식 - LoRA 어댑터만 저장
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            
            try:
                mm_projector_state = {}
                for name, param in model.base_model.model.model.mm_projector.named_parameters():
                    mm_projector_state[f'mm_projector.{name}'] = maybe_zero_3(param, ignore_status=True)
                torch.save(mm_projector_state, os.path.join(training_args.output_dir, 'mm_projector.bin'))
                print(f"Saved mm_projector to {os.path.join(training_args.output_dir, 'mm_projector.bin')}")
            except:
                print("No mm_projector found, skipping save.")
            
            try:
                alignment_projector_state = {}
                for name, param in model.alignment_projector.named_parameters():
                    alignment_projector_state[f'alignment_projector.{name}'] = maybe_zero_3(param, ignore_status=True)
                torch.save(alignment_projector_state, os.path.join(training_args.output_dir, 'alignment_projector.bin'))
                print(f"Saved alignment_projector to {os.path.join(training_args.output_dir, 'alignment_projector.bin')}")
            except:
                print("No alignment_projector found, skipping save.")
                
            try:
                denoiser_state = {}
                for name, param in model.base_model.model.model.denoiser.named_parameters():
                    denoiser_state[f'denoiser.{name}'] = maybe_zero_3(param, ignore_status=True)
                torch.save(denoiser_state, os.path.join(training_args.output_dir, 'denoiser.bin'))
                print(f"Saved denoiser to {os.path.join(training_args.output_dir, 'denoiser.bin')}")
            except:
                print("No denoiser found, skipping save.")
        
                
            
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
