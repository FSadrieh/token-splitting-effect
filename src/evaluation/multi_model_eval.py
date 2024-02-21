import argparse
from lightning import seed_everything
import csv

from utils import get_model_names_from_numbers, create_trainer_etc

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from src.training.model import BasicLM  # noqa: E402


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_name", type=str)
    parser.add_argument("model_numbers", type=str)
    parser.add_argument("config", type=str, help="path to the config file for the validation")
    parser.add_argument("-a", "--accelerator", type=str, default="cuda", help="Supports: cuda, cpu, tpu, mps")
    parser.add_argument("-p", "--prompt_length", type=int, default=16)
    return parser.parse_args()


def main():
    seed_everything(workers=True, seed=42)

    args = arg_parser()
    # We want to validate the modle on each of the 25 models
    model_names = get_model_names_from_numbers(range(25))
    model_args, dm, trainer = create_trainer_etc(args.config, model_names[0], args.accelerator, args.prompt_length)
    model_args["local_soft_prompt"] = f"logs/explainable-soft-prompts/{args.soft_prompt_name}/checkpoints/soft_prompt.pt"
    val_losses = []
    for model_name in model_names:
        model_args["model_names_or_paths"] = [model_name]
        model = BasicLM(**model_args)

        print(f"Validating {args.soft_prompt_name} on {model_name}")
        val_losses.append(trainer.validate(model, dm)[0]["val/loss"])

    with open(f"logs/explainable-soft-prompts/{args.soft_prompt_name}/checkpoints/val_losses.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "val_loss", "trained_on"])
        for i, val_loss in enumerate(val_losses):
            writer.writerow([model_names[i], val_loss, "Yes" if i in args.model_numbers.split(",") else "No"])


if __name__ == "__main__":
    main()
