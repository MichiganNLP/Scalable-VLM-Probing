#!/usr/bin/env python
import argparse
import os
import random
from typing import Any, MutableMapping

import numpy as np
import requests
import torch
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor, ProcessorMixin

from argparse_with_defaults import ArgumentParserWithDefaults

Instance = MutableMapping[str, Any]


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--dataset", default="red_caps",
                        help="See options at https://huggingface.co/datasets?"
                             "task_categories=task_categories:image-to-text")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int,
                        default=len(os.sched_getaffinity(0)) // max(torch.cuda.device_count(), 1))

    parser.add_argument("--model-name-or-path", default="openai/clip-vit-large-patch14",
                        help="See options at https://huggingface.co/models?pipeline_tag=zero-shot-image-classification")

    parser.add_argument("--output-path", default="output.pt")
    args = parser.parse_args()

    args.device = torch.device(args.device)

    return args


def fetch_image(instance: Instance) -> Instance:
    instance["image"] = Image.open(requests.get(instance["image_url"], stream=True).raw)
    return instance


def preprocess_data(processor: ProcessorMixin, instance: Instance) -> Instance:
    return processor(text=instance["caption"], images=instance["image"], return_tensors="pt")  # noqa


def main() -> None:
    args = parse_args()

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.use_deterministic_algorithms(True)
    # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    dataset = load_dataset(args.dataset, split="train", streaming=True)
    dataset = dataset.filter(input_columns=["image_url", "caption"])
    dataset = dataset.map(fetch_image)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path).to(args.device).eval()

    dataset = dataset.map(lambda instance: preprocess_data(processor, instance))
    dataset = dataset.with_format("torch")
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=args.device.type == "cuda")

    scores = []

    with torch.inference_mode():
        for batch in tqdm(data_loader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            output = model(**batch)
            scores.append(output.logits.cpu())

    torch.save(scores, args.output_path)


if __name__ == "__main__":
    main()
