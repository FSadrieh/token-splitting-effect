import argparse
from simple_parsing import parse_known_args
from transformers import AutoTokenizer
from lightning import Trainer, seed_everything
import csv
import torch
from typing import List

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from utils import get_model_names_from_numbers # noqa: E402	
from src.training.model import BasicLM # noqa: E402
from src.training.data_loading import LMDataModule # noqa: E402
from args import TrainingArgs # noqa: E402


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_name", type=str)
    parser.add_argument("model_numbers", type=str, help="Comma separated list of model numbers to test on.")
    parser.add_argument("config", type=str, help="path to the config file for the validation")
    parser.add_argument(
        "dropped_out_tokens",
        type=str,
        help="Comma separated list of tokens to drop out. If you want to drop out all tokens one by one, use 'all'. If all is set we will save a csv with all the losses for all the tokens.",
    )
    parser.add_argument(
        "-i",
        "--inverse",
        action="store_true",
        help="If set we drop out all tokens except th dropped_out_tokens. Ignored if dropped_out_tokens is 'all'.",
    )
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
        args.dropped_out_tokens,
        args.inverse,
    )
    print(model_per_token)


def prompt_token_drop_out(
    model_numbers: str,
    config: str,
    soft_prompt_name: str,
    accelerator: str,
    prompt_length: int,
    embedding_size: int,
    dropped_out_tokens: str = "all",
    inverse: bool = False,
) -> List[int]:
    seed_everything(workers=True, seed=42)

    model_names = get_model_names_from_numbers(model_numbers.split(","))

    weights = create_soft_prompt_weights(soft_prompt_name, dropped_out_tokens, prompt_length, embedding_size, inverse)
    model_args, dm, trainer = create_trainer_etc(config, model_names, accelerator, prompt_length)

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
        f"logs/explainable-soft-prompts/{soft_prompt_name}/checkpoints/{model_numbers.replace(',','_')}_with_drop_out{dropped_out_tokens}_and_inv_{inverse}.csv",
        "w+",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["seed"] + [i for i in range(len(weights))])
        for i in range(len(model_names)):
            writer.writerow([model_names[i]] + [val_losses[(model_names[i], j)] for j in range(len(weights))])

    # We return the model with the highest loss for each token
    # TODO: CHECK IF THIS IS CORRECT
    return [max(val_losses, key=val_losses.get)[1] for i in range(prompt_length)]



def create_soft_prompt_weights(
    soft_prompt_name: str, dropped_out_tokens: str, prompt_length: int, embedding_size: int, inverse: bool
) -> List[torch.nn.Parameter]:
    soft_prompt_path = f"logs/explainable-soft-prompts/{soft_prompt_name}/checkpoints/soft_prompt.pt"
    soft_prompt_weight = torch.load(soft_prompt_path)["weight"]

    if dropped_out_tokens == "all":
        weights = []
        for i in range(prompt_length):
            weight = soft_prompt_weight.detach().clone()
            weight[i, :] = torch.zeros(embedding_size)
            weights.append(torch.nn.Parameter(weight))
        return weights

    dropped_out_tokens = dropped_out_tokens.split(",")
    if inverse:
        dropped_out_tokens = [i for i in range(prompt_length) if str(i) not in dropped_out_tokens]
    print("Dropped out tokens:", dropped_out_tokens)
    for i in dropped_out_tokens:
        soft_prompt_weight[int(i), :] = torch.zeros(embedding_size)
    return [torch.nn.Parameter(soft_prompt_weight)]


def create_trainer_etc(config: str, model_names: List[str], accelerator: str, prompt_length: int):
    training_args, __ = parse_known_args(TrainingArgs, config_path=config)

    tokenizer = AutoTokenizer.from_pretrained(model_names[0], use_fast=True)

    model_args = dict(
        tokenizer=tokenizer,
        from_scratch=training_args.from_scratch,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        beta1=training_args.beta1,
        beta2=training_args.beta2,
        lr_schedule=training_args.lr_schedule,
        warmup_period=training_args.warmup_period,
        prompt_length=training_args.prompt_length,
        init_text=training_args.init_text,
        init_embedding_models=training_args.init_embedding_models,
        init_embedding_mode=training_args.init_embedding_mode,
        init_seed=training_args.init_seed,
    )

    dm = LMDataModule(training_args=training_args, tokenizer=tokenizer, prompt_length=prompt_length)

    trainer = Trainer(
        max_epochs=training_args.training_goal,
        devices=training_args.num_devices,
        accelerator=accelerator,
        strategy=training_args.distributed_strategy,
        deterministic=training_args.force_deterministic,
        precision=training_args.precision,
        gradient_clip_val=training_args.grad_clip,
        inference_mode=not training_args.compile,  # inference_mode for val/test and PyTorch 2.0 compiler don't like each other
    )

    return model_args, dm, trainer


if __name__ == "__main__":
    main()
