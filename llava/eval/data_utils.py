"""
Helpful functions for evaluation.
"""
import torch
import os
import json

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# General Dataset class for MSCOCO-CHAIR, AMBER-G/D, Haloquest, POPE, and others
class CommonDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(add_generation_prompt=True)

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        metadata = {
            "question_id": line["question_id"],
            "question": line["text"],
            "image_id": line["image"]
        }

        return input_ids, image_tensor, image.size, metadata

    def __len__(self):
        return len(self.questions)


# Specialized Dataset class for MM-HAL
class MMHalDataset(Dataset):
    def __init__(self, hg_dataset, tokenizer, image_processor, model_config, conv_mode):
        self.dataset = hg_dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        sample = self.dataset[index]
        image_file = sample["image_path"]
        qs = sample["question"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(add_generation_prompt=True)

        image = Image.open(image_file).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        metadata = {
            "question_id": sample["id"],
            "question": sample["question"],
            "image_id": sample["image_path"]
        }

        return input_ids, image_tensor, image.size, metadata

    def __len__(self):
        return len(self.dataset)


# Specialized Dataset class for MME-Evaluation
class MMEDataset(Dataset):
    def __init__(self, hg_dataset, tokenizer, image_processor, model_config, conv_mode):
        self.dataset = hg_dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        sample = self.dataset[index]
        img = sample["image"]
        qs = sample["question"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(add_generation_prompt=True)

        image = img.convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        metadata = {
            "question_id": sample["question_id"],
            "question": sample["question"],
            "image_id": sample["image"]
        }

        return input_ids, image_tensor, image.size, metadata

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, metadatas = zip(*batch)

    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes, metadatas


def get_dataloader(args, tokenizer, image_processor, model_config):
    if args.dataset == "MME":
        hg_dataset = load_dataset(args.question_file, data_dir="data")["test"]
        dataset = [sample for sample in hg_dataset]
    elif args.dataset == "MMHAL":
        hg_dataset = load_dataset(args.question_file)['test']
        dataset = [sample for sample in hg_dataset]
    else:
        dataset = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    dataset = get_chunk(dataset, args.num_chunks, args.chunk_idx)

    if args.dataset == "MME":
        dataset = MMEDataset(dataset, tokenizer, image_processor, model_config, args.conv_mode)
    elif args.dataset == "MMHAL":
        dataset = MMHalDataset(dataset, tokenizer, image_processor, model_config, args.conv_mode)
    else:
        dataset = CommonDataset(dataset, args.image_folder, tokenizer, image_processor, model_config, args.conv_mode)

    data_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)
    return data_loader

