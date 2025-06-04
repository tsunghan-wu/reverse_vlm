"""
Helpful functions for evaluation.
"""
import torch
import os
import json

from qwenvl.train.constants import LLAVA_IMAGE_TOKEN
from torch.utils.data import Dataset, DataLoader
from qwen_vl_utils import process_vision_info
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
    def __init__(self, questions, image_folder, processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.processor = processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": os.path.join(self.image_folder, image_file)},
                    {"type": "text", "text": qs}
                ]
            }
        ]
        # , "image": os.path.join(self.image_folder, image_file)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        inputs = self.processor(
            text=[text],
            images=[image_inputs],
            video_inputs=None,
            padding=True,
            return_tensors="pt",
        )
        metadata = {
            "question_id": line["question_id"],
            "question": line["text"],
            "image_id": line["image"]
        }

        return inputs, metadata

    def __len__(self):
        return len(self.questions)


# Specialized Dataset class for MM-HAL
class MMHalDataset(Dataset):
    def __init__(self, hg_dataset, processor, model_config):
        self.dataset = hg_dataset
        self.processor = processor
        self.model_config = model_config

    def __getitem__(self, index):
        sample = self.dataset[index]
        image_file = sample["image_path"]
        qs = sample["question"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_file},
                    {"type": "text", "text": qs}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        metadata = {
            "question_id": sample["id"],
            "question": sample["question"],
            "image_id": sample["image_path"]
        }

        return inputs, metadata

    def __len__(self):
        return len(self.dataset)


# Specialized Dataset class for MME-Evaluation
class MMEDataset(Dataset):
    def __init__(self, hg_dataset, processor, model_config):
        self.dataset = hg_dataset
        self.processor = processor
        self.model_config = model_config

    def __getitem__(self, index):
        sample = self.dataset[index]
        img = sample["image"]
        qs = sample["question"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": qs}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors="pt",
        )
        metadata = {
            "question_id": sample["question_id"],
            "question": sample["question"],
            "image_id": sample["image"]
        }
        return inputs, metadata

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    # return batch
    inputs, metadatas = zip(*batch)
    assert len(inputs) == 1, "Batch size should be 1 for this dataset"
    # input_ids = torch.stack(input_ids, dim=0)
    # image_tensors = torch.stack(image_tensors, dim=0)
    return inputs, metadatas


def get_dataloader(args, processor, model_config):
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
        dataset = MMEDataset(dataset, processor, model_config)
    elif args.dataset == "MMHAL":
        dataset = MMHalDataset(dataset, processor, model_config)
    else:
        dataset = CommonDataset(dataset, args.image_folder, processor, model_config)

    data_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)
    return data_loader

