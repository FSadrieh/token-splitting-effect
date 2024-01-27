import argparse
from typing import List
import torch

from utils import get_model_names_from_numbers, create_trainer_etc, load_soft_prompt_weight

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from src.training.model import BasicLM  # noqa: E402


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_name", type=str)
    parser.add_argument("model_number", type=int)
    parser.add_argument("config", type=str, help="path to the config file for the validation")
    parser.add_argument(
        "dropped_out_tokens",
        type=str,
        help="Comma separated list of tokens to drop out.",
    )
    parser.add_argument(
        "-i",
        "--inverse",
        action="store_true",
        help="If set we drop out all tokens except the dropped_out_tokens.",
    )
    parser.add_argument(
        "-s",
        "--shorten",
        action="store_true",
        help="If set the token is cut, thus reducing the prompt length.",
    )
    parser.add_argument("-a", "--accelerator", type=str, default="cuda", help="Supports: cuda, cpu, tpu, mps")
    parser.add_argument("-p", "--prompt_length", type=int, default=16)
    parser.add_argument("-e", "--embedding_size", type=int, default=768)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    return parser.parse_args()


def main():
    args = arg_parser()
    dropped_out_tokens = args.dropped_out_tokens.split(",")
    loss = validate_prompt(
        args.model_number,
        args.config,
        args.soft_prompt_name,
        args.accelerator,
        args.prompt_length,
        args.embedding_size,
        dropped_out_tokens,
        args.inverse,
        args.shorten,
        args.batch_size,
    )
    print(loss)


def validate_prompt(
    model_number: int,
    config: str,
    soft_prompt_name: str,
    accelerator: str,
    prompt_length: int,
    embedding_size: int,
    dropped_out_tokens: List[int],
    inverse: bool,
    shorten: bool,
    batch_size: int,
) -> float:
    model_name = get_model_names_from_numbers(model_number)[0]

    weight, new_prompt_length = create_soft_prompt_weights(
        soft_prompt_name, dropped_out_tokens, prompt_length, embedding_size, inverse, shorten
    )
    model_args, dm, trainer = create_trainer_etc(config, model_name, accelerator, new_prompt_length, batch_size)
    model_args["model_names_or_paths"] = [model_name]
    model = BasicLM(**model_args)
    model.set_soft_prompt_weight(weight)

    return trainer.validate(model, dm)[0]["val/loss"]


def create_soft_prompt_weights(
    soft_prompt_name: str, dropped_out_tokens: List[int], prompt_length: int, embedding_size: int, inverse: bool, shorten: bool
) -> (torch.nn.Parameter, int):
    soft_prompt_weight = load_soft_prompt_weight(soft_prompt_name)
    if inverse:
        dropped_out_tokens = [i for i in range(prompt_length) if i not in dropped_out_tokens]
    print("Dropped out tokens:", dropped_out_tokens)

    if shorten:
        new_prompt_length = prompt_length - len(dropped_out_tokens)
        new_weight = torch.zeros(prompt_length, embedding_size)
        for i in range(prompt_length):
            if i not in dropped_out_tokens:
                new_weight[i, :] = soft_prompt_weight[i, :]
        return torch.nn.Parameter(new_weight), new_prompt_length

    for i in dropped_out_tokens:
        soft_prompt_weight[i, :] = torch.zeros(embedding_size)
    return torch.nn.Parameter(soft_prompt_weight), prompt_length


if __name__ == "__main__":
    main()
