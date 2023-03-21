#!/usr/bin/env python
import argparse
import logging
import math
import os
import random
from typing import Any, MutableMapping

import numpy as np
import pandas as pd
import torch
from PIL import Image
from cached_path import cached_path
from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor, ProcessorMixin
from transformers.models.clip.modeling_clip import CLIPModel

from argparse_with_defaults import ArgumentParserWithDefaults

Instance = MutableMapping[str, Any]


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=torch.device, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--dataset", default="red_caps",
                        help="See options at https://huggingface.co/datasets?"
                             "task_categories=task_categories:image-to-text")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-streaming-mode", default="map", choices=["load", "map", "none"],
                        help="Streaming mode for the dataset. \"load\" means that the dataset is in streaming mode when"
                             " loaded, \"map\" means that the dataset is in streaming mode before any map function is"
                             " applied, and \"none\" means that the dataset is not in streaming mode.\n"
                             " This is useful for large datasets that don't fit in"
                             " memory (the captions; because the images are downloaded on demand) or that take time to"
                             " download. Note that, in streaming mode, the dataset is shuffled by relatively small"
                             " buffers, instead of being completely shuffled. This can lead to a certain dataset order,"
                             " and mess with statistical properties if only a portion of the dataset is used in this"
                             " mode (i.e., you're not taking a random sample of the dataset).\n"
                             " Thus, is recommended to avoid streaming mode if the dataset is small. If it's not small,"
                             " you should use streaming mode when loading if you know it has a random order (supposing"
                             " you care about a random sample), otherwise you should use the streaming mode when"
                             " mapping (by shuffling and taking a subset before downloading the images).")

    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle to avoid a certain dataset order, for statistical reasons.")

    parser.add_argument("--max-examples", type=int)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int,
                        default=len(os.sched_getaffinity(0)) // max(torch.cuda.device_count(), 1))

    parser.add_argument("--model-name-or-path", default="openai/clip-vit-large-patch14",
                        help="See options at https://huggingface.co/models?pipeline_tag=zero-shot-image-classification")

    parser.add_argument("--output-path", default="output.csv")

    parser.add_argument("--logging-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    return parser.parse_args()


def set_deterministic_mode(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def fetch_image(instance: Instance) -> Instance:
    try:
        return {"image": Image.open(cached_path(instance["image_url"], quiet=True))}
    except FileNotFoundError:
        return {"image": None}


def preprocess_data(processor: ProcessorMixin, instance: Instance) -> Instance:
    return processor(text=instance["caption"], images=instance["image"], truncation=True,  # noqa
                     padding=True, return_tensors="pt")


def compute_scores(model: CLIPModel, batch: Instance) -> torch.Tensor:
    output = model(**batch, return_dict=True)
    return (output.text_embeds * output.image_embeds).sum(dim=-1)


def main() -> None:
    args = parse_args()

    print(args)

    logging.getLogger().setLevel(args.logging_level.upper())

    set_deterministic_mode(args.seed)

    # We tokenize the text in each data loading worker, which in turn is supposed to run in its own process.
    # So we disable the HuggingFace fast tokenizer parallelism (if available) because we're already doing parallelism
    # at the data loading level. It should be more efficient this way.
    os.environ["TOKENIZERS_PARALLELISM"] = "0"

    dataset = load_dataset(args.dataset, split=args.dataset_split, streaming=args.dataset_streaming_mode == "load",
                           num_proc=None if args.dataset_streaming_mode == "load" else args.num_workers)
    dataset = dataset.select_columns(["image_id", "caption", "image_url"])

    num_examples = dataset.info.splits[args.dataset_split].num_examples

    if args.shuffle:
        dataset = dataset.shuffle()

    if args.max_examples:
        num_examples = min(args.max_examples, num_examples)
        if isinstance(dataset, IterableDataset):
            dataset = dataset.take(num_examples)
        else:
            dataset = dataset.select(range(num_examples))

    if args.dataset_streaming_mode == "map":
        dataset = dataset.to_iterable_dataset(num_shards=args.num_workers)

    fetch_image_map_kwargs = {}
    if not isinstance(dataset, IterableDataset):
        fetch_image_map_kwargs["num_proc"] = args.num_workers
        fetch_image_map_kwargs["desc"] = "Downloading the images"
    dataset = dataset.map(fetch_image, remove_columns=["image_url"], **fetch_image_map_kwargs)
    dataset = dataset.filter(lambda instance: instance["image"] is not None)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    preprocess_data_map_kwargs = {}
    if not isinstance(dataset, IterableDataset):
        preprocess_data_map_kwargs["num_proc"] = args.num_workers
        preprocess_data_map_kwargs["desc"] = "Preprocessing the data"
    dataset = dataset.map(lambda instance: preprocess_data(processor, instance), batched=True,
                          batch_size=args.batch_size, remove_columns=["image"], **preprocess_data_map_kwargs)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=args.device.type != "cpu")

    model = AutoModel.from_pretrained(args.model_name_or_path).to(args.device).eval()

    image_ids = []
    text_list = []
    score_list = []

    with torch.inference_mode():
        for batch in tqdm(data_loader, total=math.ceil(num_examples / args.batch_size)):
            image_ids.extend(batch.pop("image_id"))
            text_list.extend(batch.pop("caption"))
            score_list.extend(compute_scores(model=model,
                                             batch={k: v.to(args.device) for k, v in batch.items()}).tolist())

    pd.DataFrame({"image_id": image_ids, "sentence": text_list, "score": score_list}).to_csv(args.output_path,
                                                                                             index=False)


if __name__ == "__main__":
    main()
