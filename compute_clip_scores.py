#!/usr/bin/env python
import argparse
import collections.abc
import itertools
import logging
import os
import random
import re
import sys
import traceback
from typing import Any, Iterable, MutableMapping, Tuple

import PIL
import math
import numpy as np
import pandas as pd
import torch
from PIL import Image
from cached_path import cached_path
from datasets import IterableDataset, load_dataset
from requests import HTTPError
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate_fn_map
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
    parser.add_argument("--dataset-files")
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
                        default=len(os.sched_getaffinity(0)) // max(torch.cuda.device_count(), 1),
                        help="Number of workers used where parallelization is allowed. Zero means the default value"
                             " (generally it means running from the main thread).")

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


def get_imgur_urls_maybe(url: str) -> Iterable[str]:
    if "imgur.com" in url:
        for part in url.split(","):
            if part:
                part = part.strip()

                root, ext = os.path.splitext(part)
                root = root if root.startswith("http") else ("http://i.imgur.com/" + root)  # noqa
                root = root.split("#", maxsplit=1)[0]
                ext = ext or ".jpg"
                ext = re.split(r"[?#]", ext, maxsplit=1)[0]

                yield root + ext
    else:
        yield url


def get_image_urls(instance: Instance) -> Iterable[str]:
    url = instance.get("image_url") or instance["pos_url"]
    for image_url in re.findall(r"http\S+", url) or [url]:
        yield from get_imgur_urls_maybe(image_url)


def fetch_image(instance: Instance) -> Instance:
    image_url = next(iter(get_image_urls(instance)))

    try:
        image = Image.open(cached_path(image_url, quiet=True))
    except (FileNotFoundError, HTTPError, PIL.UnidentifiedImageError):
        image = None
    except Exception as e:  # noqa
        print("Unknown error while fetching an image:", file=sys.stderr)
        traceback.print_exc()
        print("The image will be skipped.")
        image = None

    return {"image": image}


def preprocess_data(processor: ProcessorMixin, instance: Instance) -> Instance:
    text = instance.get("caption") or instance["sentence"]
    return processor(text=text, images=instance["image"], truncation=True,  # noqa
                     padding=True, return_tensors="pt")


def get_non_collatable_columns(instance: Instance) -> Iterable[str]:
    for k, v in instance.items():
        if not (any(isinstance(v, types) for types in default_collate_fn_map)
                or isinstance(v, (collections.abc.Sequence, collections.abc.Mapping))):
            yield k


def compute_scores(model: CLIPModel, batch: Instance) -> torch.Tensor:
    output = model(**batch, return_dict=True)
    return (output.text_embeds * output.image_embeds).sum(dim=-1)


def zip_equal(*iterables: Iterable[Any]) -> Iterable[Tuple[Any, ...]]:
    sentinel = object()
    for combo in itertools.zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def save_output(batches: Iterable[Instance], path: str) -> None:
    pd.DataFrame(batches).to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    print(args)

    logging.getLogger().setLevel(args.logging_level.upper())

    set_deterministic_mode(args.seed)

    # We tokenize the text in each data loading worker, which in turn is supposed to run in its own process.
    # So we disable the HuggingFace fast tokenizer parallelism (if available) because we're already doing parallelism
    # at the data loading level. It should be more efficient this way.
    os.environ["TOKENIZERS_PARALLELISM"] = "0"

    data_files = {args.dataset_split: args.dataset_files} if args.dataset_files else None
    dataset = load_dataset(args.dataset, split=args.dataset_split, data_files=data_files,
                           streaming=args.dataset_streaming_mode == "load",
                           num_proc=None if args.dataset_streaming_mode == "load" else (args.num_workers or None))

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
        dataset = dataset.to_iterable_dataset(num_shards=args.num_workers or 1)

    fetch_image_map_kwargs = {}
    if not isinstance(dataset, IterableDataset):
        fetch_image_map_kwargs["num_proc"] = args.num_workers or None
        fetch_image_map_kwargs["desc"] = "Downloading the images"
    dataset = dataset.map(fetch_image, **fetch_image_map_kwargs)
    dataset = dataset.filter(lambda instance: instance["image"] is not None)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    preprocess_data_map_kwargs = {}
    if not isinstance(dataset, IterableDataset):
        preprocess_data_map_kwargs["num_proc"] = args.num_workers or None
        preprocess_data_map_kwargs["desc"] = "Preprocessing the data"
    dataset = dataset.map(lambda instance: preprocess_data(processor, instance), batched=True,
                          batch_size=args.batch_size, remove_columns=["image"], **preprocess_data_map_kwargs)

    dataset = dataset.remove_columns(list(get_non_collatable_columns(next(iter(dataset)))))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=args.device.type != "cpu")

    print("Loading the model…")
    model = AutoModel.from_pretrained(args.model_name_or_path).to(args.device).eval()
    print("Model loaded.")

    batches = []

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(data_loader, total=math.ceil(num_examples / args.batch_size))):
            model_inputs = {k: batch.pop(k).to(args.device)
                            for k in list(batch.keys())  # We need to copy the keys because we're modifying the dict.
                            if k in {"input_ids", "attention_mask", "pixel_values"}}
            batch["clip_score"] = compute_scores(model=model, batch=model_inputs).tolist()

            batch = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Go from a dict of lists to a "list" of dicts:
            batches.extend(dict(zip_equal(batch.keys(), instance_tuple))  # noqa
                           for instance_tuple in zip_equal(*batch.values()))

            if i % 1000 == 1:  # Save the data occasionally in case there's a crash.
                save_output(batches, path=args.output_path)

    save_output(batches, path=args.output_path)


if __name__ == "__main__":
    main()
