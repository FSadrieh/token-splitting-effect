import argparse
import csv
import torch
from typing import List
import numpy as np

from utils import get_model_names_from_numbers, create_trainer_etc, load_soft_prompt_weight

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from src.training.model import BasicLM  # noqa: E402


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_name", type=str)
    parser.add_argument("model_numbers", type=str, help="Comma separated list of model numbers to test on.")
    parser.add_argument("config", type=str, help="path to the config file for the validation")
    parser.add_argument("-a", "--accelerator", type=str, default="cuda", help="Supports: cuda, cpu, tpu, mps")
    parser.add_argument("-p", "--prompt_length", type=int, default=16)
    parser.add_argument("-e", "--embedding_size", type=int, default=768)
    return parser.parse_args()


def main():
    args = arg_parser()
    model_per_token = prompt_token_drop_out(
        args.model_numbers,
        args.config,
        args.soft_prompt_name,
        args.accelerator,
        args.prompt_length,
        args.embedding_size,
    )
    print(model_per_token)


def prompt_token_drop_out(
    model_numbers: str,
    config: str,
    soft_prompt_name: str,
    accelerator: str,
    prompt_length: int,
    embedding_size: int,
) -> List[int]:
    model_names = get_model_names_from_numbers(model_numbers.split(","))

    weights = create_soft_prompt_weights(soft_prompt_name, prompt_length, embedding_size)
    model_args, dm, trainer = create_trainer_etc(config, model_names[0], accelerator, prompt_length)

    val_losses = {}
    # We validate for each model and each weight
    for model_name in model_names:
        model_args["model_names_or_paths"] = [model_name]
        model = BasicLM(**model_args)
        for i, weight in enumerate(weights):
            model.set_soft_prompt_weight(weight)
            print(f"Validating {soft_prompt_name} on {model_name}. The token {i} is dropped out.")
            val_losses[(model_name, i)] = trainer.validate(model, dm)[0]["val/loss"]

    # We save all losses in a csv file at the location of the soft prompt
    with open(
        f"logs/explainable-soft-prompts/{soft_prompt_name}/checkpoints/{model_numbers.replace(',','_')}_token_drop_out.csv",
        "w+",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["seed"] + [i for i in range(len(weights))])
        for i in range(len(model_names)):
            writer.writerow([model_names[i]] + [val_losses[(model_names[i], j)] for j in range(len(weights))])

    # We return the model with the highest loss for each token
    return [np.argmax([val_losses[(model_names[j], i)] for j in range(len(model_names))]) for i in range(len(weights))]


def create_soft_prompt_weights(soft_prompt_name: str, prompt_length: int, embedding_size: int) -> List[torch.nn.Parameter]:
    soft_prompt_weight = load_soft_prompt_weight(soft_prompt_name)
    weights = []
    for i in range(prompt_length):
        weight = soft_prompt_weight.detach().clone()
        weight[i, :] = torch.zeros(embedding_size)
        weights.append(torch.nn.Parameter(weight))
    return weights


if __name__ == "__main__":
    main()
