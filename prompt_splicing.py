import argparse
from simple_parsing import parse_known_args
from transformers import AutoTokenizer
from lightning import Trainer, seed_everything
import csv
import torch


from src.evaluation.utils import get_model_names_from_numbers
from src.training.model import BasicLM
from src.training.data_loading import LMDataModule
from args import TrainingArgs


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_name", type=str)
    parser.add_argument("model_numbers", type=str, help="Comma separated list of model numbers to test on.")
    parser.add_argument("config", type=str, help="path to the config file for the validation")
    parser.add_argument("dropped_out_tokens", type=str, help="Comma separated list of tokens to drop out")
    parser.add_argument("-a", "--accelerator", type=str, default="cuda", help="Supports: cuda, cpu, tpu, mps")
    parser.add_argument("-p", "--prompt_length", type=int, default=16)
    parser.add_argument("-e", "--embedding_size", type=int, default=768)
    return parser.parse_args()


def main():
    seed = seed_everything(workers=True, seed=42)

    args = arg_parser()

    soft_prompt_path = f"logs/explainable-soft-prompts/{args.soft_prompt_name}/checkpoints/soft_prompt.pt"
    soft_prompt = torch.nn.Embedding(args.prompt_length, args.embedding_size)
    soft_prompt.load_state_dict(torch.load(soft_prompt_path))

    weigth = soft_prompt.weight.detach().clone()
    for dropped_out_token in args.dropped_out_tokens.split(","):
        weigth[int(dropped_out_token), :] = torch.zeros(args.embedding_size)

    soft_prompt.weight = torch.nn.Parameter(weigth)

    model_names = get_model_names_from_numbers(args.model_numbers.split(","))
    tokenizer = AutoTokenizer.from_pretrained(model_names[0], use_fast=True)
    val_losses = []
    for model_name in model_names:
        training_args, __ = parse_known_args(TrainingArgs, config_path=args.config)
        model_args = dict(
            model_names_or_paths=[model_name],
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
            local_soft_prompt=soft_prompt,
        )
        model = BasicLM(**model_args)

        dm = LMDataModule(training_args=training_args, tokenizer=tokenizer, prompt_length=args.prompt_length)

        trainer = Trainer(
            max_epochs=training_args.training_goal,
            devices=training_args.num_devices,
            accelerator=args.accelerator,
            strategy=training_args.distributed_strategy,
            deterministic=training_args.force_deterministic,
            precision=training_args.precision,
            gradient_clip_val=training_args.grad_clip,
            inference_mode=not training_args.compile,  # inference_mode for val/test and PyTorch 2.0 compiler don't like each other
        )

        print(f"Validating {args.soft_prompt_name} on {model_name}")
        val_losses.append(trainer.validate(model, dm)[0]["val/loss"])

    with open(f"logs/explainable-soft-prompts/{args.soft_prompt_name}/checkpoints/prompt_sclicing.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "val_loss", "trained_on"])
        for i, val_loss in enumerate(val_losses):
            writer.writerow([model_names[i], val_loss, "Yes" if i in args.model_numbers.split(",") else "No"])


if __name__ == "__main__":
    main()
