#!/usr/bin/env python
import argparse
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

from argparse_with_defaults import ArgumentParserWithDefaults


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="red_caps",
                        help="See options at https://huggingface.co/datasets?"
                             "task_categories=task_categories:image-to-text")
    parser.add_argument("--model-name-or-path", default="openai/clip-vit-large-patch14",
                        help="See options at https://huggingface.co/models?pipeline_tag=zero-shot-image-classification")
    parser.add_argument("--output-path", default="output.pt")
    return parser.parse_args()

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

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path).to(DEVICE).eval()

    scores = []

    with torch.inference_mode():
        for batch in tqdm(load_dataset(args.dataset, split="train", streaming=True)):
            batch = batch.to(DEVICE)
            output = model(**batch)
            scores.append(output.logits.cpu())

    torch.save(scores, args.output_path)


if __name__ == "__main__":
    main()
