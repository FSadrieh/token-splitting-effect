import argparse
from typing import List, Tuple
import torch

from utils import load_soft_prompt_weight, validate_soft_prompt, create_trainer_etc, get_model_names_from_numbers


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
    use_test_set: bool = False,
    seed: int = None,
    number_of_tokens_to_drop: int = None,  # Only used if seed is set
) -> float:
    """
    validate_prompt() is a function that validates parts of a SP on multiple models. With dropped out tokens one can specify which tokens to drop out. Inverse makes dropped out tokens to tokens to keep.
    When short is set the prompt length is reduced by the number of dropped out tokens. Otherwise the prompt length is kept the same and the dropped out tokens are set to zero.
    If a seed is set, the dropped out tokens are chosen randomly. The number of tokens to drop is set by number_of_tokens_to_drop.
    """
    print(f"Validating {soft_prompt_name} on model {model_number}")

    if seed:
        torch.manual_seed(seed)
        dropped_out_tokens = torch.randperm(prompt_length)[:number_of_tokens_to_drop].tolist()

    weight, new_prompt_length = create_soft_prompt_weights(
        soft_prompt_name, dropped_out_tokens, prompt_length, embedding_size, inverse, shorten
    )
    model_name = get_model_names_from_numbers([model_number])[0]
    model_args, dm, trainer = create_trainer_etc(config, model_name, accelerator, new_prompt_length, batch_size)
    return validate_soft_prompt(
        model_args=model_args, trainer=trainer, dm=dm, model_number=model_number, use_test_set=use_test_set, weight=weight
    )


def create_soft_prompt_weights(
    soft_prompt_name: str, dropped_out_tokens: List[int], prompt_length: int, embedding_size: int, inverse: bool, shorten: bool
) -> Tuple[torch.nn.Parameter, int]:
    soft_prompt_weight = load_soft_prompt_weight(soft_prompt_name)
    if inverse:
        dropped_out_tokens = [i for i in range(prompt_length) if i not in dropped_out_tokens]

    if shorten:
        print("Dropped out tokens:", dropped_out_tokens)
        new_prompt_length = prompt_length - len(dropped_out_tokens)
        remaining_tokens = [i for i in range(prompt_length) if i not in dropped_out_tokens]
        new_weight = torch.zeros(new_prompt_length, embedding_size)
        for i, token_idx in enumerate(remaining_tokens):
            new_weight[i, :] = soft_prompt_weight[token_idx, :]
        return torch.nn.Parameter(new_weight), new_prompt_length

    print("Masked tokens:", dropped_out_tokens)
    for i in dropped_out_tokens:
        soft_prompt_weight[i, :] = torch.zeros(embedding_size)
    return torch.nn.Parameter(soft_prompt_weight), prompt_length


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_name", type=str)
    parser.add_argument("model_number", type=int)
    parser.add_argument("config", type=str, help="path to the config file for the validation")
    parser.add_argument(
        "--dropped_out_tokens",
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
    parser.add_argument("-t", "--use_test_set", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The seed argument overrides dropped_out_tokens and chooses a random set of tokens to drop out.",
    )
    parser.add_argument(
        "-k",
        "--number_of_tokens_to_drop",
        type=int,
        default=None,
        help="The number of tokens to drop out. Only used if seed is set.",
    )
    return parser.parse_args()


def main():
    args = arg_parser()
    if args.dropped_out_tokens:
        dropped_out_tokens = args.dropped_out_tokens.split(",")
        dropped_out_tokens = [int(i) for i in dropped_out_tokens]
    loss = validate_prompt(
        args.model_number,
        args.config,
        args.soft_prompt_name,
        args.accelerator,
        args.prompt_length,
        args.embedding_size,
        dropped_out_tokens if args.dropped_out_tokens else [],
        args.inverse,
        args.shorten,
        args.batch_size,
        args.use_test_set,
        args.seed,
        args.number_of_tokens_to_drop,
    )
    print(loss)


if __name__ == "__main__":
    main()
