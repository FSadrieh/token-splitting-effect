import argparse
import torch
from pathlib import Path
import os

from utils import load_soft_prompt_weight, load_init_text

DEFAULT_SAVE_DIR = Path("logs/explainable-soft-prompts/")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_name1", type=str)
    parser.add_argument("soft_prompt_name2", type=str)
    parser.add_argument("operator", type=str)
    parser.add_argument(
        "-i",
        "--use_initial_prompt",
        type=int,
        default=0,
        help="Use the initial prompt instead of the trained one. Specify 1 for prompt1 and 2 for prompt2.",
    )
    parser.add_argument("-e", "--embedding_size", type=int, default=768)
    parser.add_argument("-p", "--prompt_length", type=int, default=16)

    return parser.parse_args()


def prompt_algebra(prompt_1: torch.Tensor, prompt_2: torch.Tensor, operator: str) -> torch.Tensor:
    if operator == "+":
        return prompt_1 + prompt_2
    elif operator == "-":
        return prompt_1 - prompt_2
    elif operator == "*":
        return prompt_1 * prompt_2
    elif operator == "/":
        return prompt_1 / prompt_2


def main():
    args = arg_parser()
    soft_prompt_name1 = args.soft_prompt_name1
    soft_prompt_name2 = args.soft_prompt_name2
    if args.use_initial_prompt == 1:
        print(f"Computing: (inital prompt of {soft_prompt_name1}) {args.operator} {soft_prompt_name2}.")
        soft_prompt1, __ = load_init_text(soft_prompt_name1)
        soft_prompt2 = load_soft_prompt_weight(soft_prompt_name2)
        soft_prompt_name1 = f"init_{soft_prompt_name1}"
    elif args.use_initial_prompt == 2:
        print(f"Computing: {soft_prompt_name1} {args.operator} (inital prompt of {soft_prompt_name2}).")
        soft_prompt1 = load_soft_prompt_weight(soft_prompt_name1)
        soft_prompt2, __ = load_init_text(soft_prompt_name2)
        soft_prompt_name2 = f"init_{soft_prompt_name2}"
    else:
        print(f"Computing {soft_prompt_name1} {args.operator} {soft_prompt_name2}.")
        soft_prompt1 = load_soft_prompt_weight(soft_prompt_name1)
        soft_prompt2 = load_soft_prompt_weight(soft_prompt_name2)

    result = prompt_algebra(soft_prompt1, soft_prompt2, args.operator)

    # To save the state dict we need to create a Embedding
    soft_prompt = torch.nn.Embedding(args.prompt_length, args.embedding_size)
    soft_prompt.weight = torch.nn.Parameter(result)

    save_dir = os.path.join(DEFAULT_SAVE_DIR, soft_prompt_name1 + args.operator + soft_prompt_name2, "checkpoints")

    os.makedirs(save_dir)
    torch.save(soft_prompt.state_dict(), save_dir + "/soft_prompt.pt")
    print(f"Saved the result to {save_dir}.")


if __name__ == "__main__":
    main()
