"""
This script allows you to download and preprocess datasets for language modeling, specifically mc4 and cc100. You can customize it to your own needs.

Example command to download cc100 for German:
python preprocess_data.py --lg de --dataset cc100 --out_dir ./data/cc100/ --processes 8


Example command to download cc100 for German using streaming mode for HF datasets (faster, requires less RAM) and cleaning up caches:
python preprocess_data.py --lg de --dataset cc100 --out_dir ./data/cc100/ --processes 8 --stream --stream_shuffle_buffer_size 1_000_000 --conserve_disk_space

Inspiration from lit-gpt and gpt-neox.
"""

import errno
import io
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import datasets
import jsonlines
from datasets import load_dataset
from print_on_steroids import graceful_exceptions, logger
from simple_parsing import field, parse


@dataclass
class Args:
    out_dir: str = field(alias="-o")
    "Path to data directory."

    dataset: Literal["imdb", "emotion", "mnli"] = field(default="imdb")
    "HF dataset. Pile currently uses a mirror with copyrighted material removed."

    max_train_size: int = field(default=50_000_000)
    "Maximum number of train documents to write to disk. Use to limit very large datasets that you will not exhaust during training anyway. Use -1 to disable."

    dev_size: int = field(default=5_000)
    "If 0, do not construct dev set."

    test_size: int = field(default=5_000)
    "If 0, do not contruct test set."

    processes: int = field(default=4)
    "Number of processes for parallel tokenization."

    split: str = field(default="train")
    "Select percentage of dataset like so: --split=train[:50%]"

    conserve_disk_space: bool = field(default=False, alias="--disk_space")
    "Disable all HF caching and cleanup download caches to conserve disk space."

    stream: bool = field(default=False)
    "Couple with max_train_size to avoid having to load the entire dataset. Use streaming mode to load dataset and process dataset."

    stream_shuffle_buffer_size: int = field(default=100_000)
    """Buffer size for shuffling datasets before splitting in streaming mode.
    The entire buffer will be downloaded before shuffling.
    You also need to have enough RAM if you set this to a large value If -1, set to max_train_size."""

    pre_discard_factor: float = field(default=None)
    """Percentage of the dataset to discard before any processing.
    Useful for speeding up processing of huge datasets that are not fully needed.
    Not needed if you use --stream."""

    format: Literal["txt", "jsonl"] = field(default="jsonl")
    "Format to write to disk. Prefer jsonl over txt for better handling of newlines in documents and because it can be laoded much faster by HF datasets."


@graceful_exceptions()
def main(args: Args):
    logger.info(args)
    if args.max_train_size == -1:
        args.max_train_size = None
    if args.conserve_disk_space:
        # Disable caching because we write the end result to disk anyways. Intermediary caches just clutter the disk!
        logger.info("Disabling caching to conserve disk space.")
        datasets.fingerprint.disable_caching()

    output_dir = os.path.join(args.out_dir, args.dataset)

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Downloading dataset. This can take some time, so sit back and relax...")

    tmp_cache_dir = None
    if args.conserve_disk_space:
        tmp_cache_dir = os.path.join(output_dir, args.language, "tmp_download_cache")
        os.makedirs(tmp_cache_dir, exist_ok=True)

    ##### Load dataset #####
    if args.dataset == "imdb":
        dataset = load_dataset(
            "imdb",
            split=args.split,
            cache_dir=tmp_cache_dir,
            streaming=args.stream,
            num_proc=None if args.stream else args.processes,
        )
        print(dataset)
    elif args.dataset == "emotion":
        dataset = load_dataset(
            "dair-ai/emotion",
            split=args.split,
            cache_dir=tmp_cache_dir,
            streaming=args.stream,
            num_proc=None if args.stream else args.processes,
        )
    elif args.dataset == "mnli":
        dataset = load_dataset(
            "glue",
            "mnli",
            split=args.split,
            cache_dir=tmp_cache_dir,
            streaming=args.stream,
            num_proc=None if args.stream else args.processes,
        )
        
        dataset = dataset.map(lambda x: {"text": x["premise"] + ";" + x["hypothesis"]})
        dataset = dataset.remove_columns(["idx", "premise", "hypothesis"])
        print(dataset)

    ##### Split into train/dev/test #####
    logger.info("Shuffling and splitting into sets...")
    total_len = len(dataset)
    logger.info(f"Dataset len after processing: {total_len}")

    dataset = dataset.shuffle(seed=42)

    dev_test_size = args.dev_size + (args.test_size or 0)
    train_end_idx = total_len - dev_test_size
    train_paragraphs = dataset.select(range(train_end_idx))
    if args.max_train_size and len(train_paragraphs) > args.max_train_size:
        train_paragraphs = train_paragraphs.select(range(args.max_train_size))

    dev_paragraphs = dataset.select(range(train_end_idx, train_end_idx + args.dev_size))
    if args.test_size:
        test_paragraphs = dataset.select(range(train_end_idx + args.dev_size, total_len))
    logger.info(f"Example train split data: {train_paragraphs[:4]}")
    logger.info(f"len: {len(train_paragraphs)}")

    if args.conserve_disk_space:
        logger.info("Cleaning download cache")
        try:
            shutil.rmtree(tmp_cache_dir)
        except OSError as e:
            # Reraise unless ENOENT: No such file or directory
            # (ok if directory has already been deleted)
            if e.errno != errno.ENOENT:
                raise

    ##### Write to disk #####
    logger.info("Writing data...")
    output_dir = Path(output_dir)
    os.makedirs(str(output_dir), exist_ok=True)
    PERFORMANT_BUFFER_SIZE_BYTES = 1024 * 1024 * 100  # 100 MB

    # Preferred.
    if args.format == "jsonl":
        train_fp = io.open(str(output_dir / "train.jsonl"), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES)
        with jsonlines.Writer(train_fp, compact=True) as writer:
            writer.write_all(train_paragraphs)
        train_fp.close()

        if args.dev_size:
            dev_fp = io.open(str(output_dir / "dev.jsonl"), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES)
            with jsonlines.Writer(dev_fp, compact=True) as writer:
                writer.write_all(dev_paragraphs)
            dev_fp.close()

        if args.test_size:
            test_fp = io.open(str(output_dir / "test.jsonl"), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES)
            with jsonlines.Writer(test_fp, compact=True) as writer:
                writer.write_all(test_paragraphs)
            test_fp.close()

    logger.success("Done! Enjoy your data :)")
    logger.print(output_dir / "train.jsonl")


if __name__ == "__main__":
    args = parse(Args)
    main(args)
