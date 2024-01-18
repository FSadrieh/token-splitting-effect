import argparse
import torch

from utils import create_soft_prompt


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_names", type=str)
    parser.add_argument("-e", "--embedding_size", type=int, default=768)
    parser.add_argument("-p", "--prompt_length", type=int, default=16)

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()

    print(f"Computing the length of {args.soft_prompt_names}:")
    for soft_prompt_name in args.soft_prompt_names.split(","):
        soft_prompt = create_soft_prompt(soft_prompt_name, args.prompt_length, args.embedding_size)
        soft_prompt = torch.mean(soft_prompt, dim=0)
        print(soft_prompt_name, torch.linalg.vector_norm(soft_prompt).item())
