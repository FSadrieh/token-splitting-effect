import argparse
import matplotlib.pyplot as plt
import datasets
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset statistics")
    parser.add_argument("--dataset", type=str, help="dataset name to get the statistics for", default="data/imdb/train.jsonl")
    parser.add_argument("--max_length", "-m", type=int, default=512, help="Maximum length of the reviews")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file for the statistics")
    return parser.parse_args()


def main():
    tokenizer = AutoTokenizer.from_pretrained("google/multiberts-seed_0", use_fast=True)
    args = parse_args()
    data_files = {"train": "data/imdb/train.jsonl", "val": "data/imdb/dev.jsonl"}
    train_val_datasets = datasets.load_dataset(
        "json",
        data_files=data_files,
        name="imdb",
        num_proc=4,
    )
    print(f"Dataset: {train_val_datasets}")
    lengths = []
    for example in train_val_datasets["train"]:
        lengths.append(len(tokenizer(example["text"])["input_ids"]))

    print(f"Number of examples: {len(train_val_datasets['train'])}")
    print(f"Average length: {sum(lengths) / len(train_val_datasets['train'])}")
    print(f"Median length: {sorted(lengths)[len(lengths) // 2]}")
    print(f"Max length: {max(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(f"Number of reviews with length > {args.max_length}: {len([l for l in lengths if l > args.max_length])}")
    if args.output:
        import seaborn as sns

        sns.displot(lengths)
        plt.savefig(args.output)


if __name__ == "__main__":
    main()
