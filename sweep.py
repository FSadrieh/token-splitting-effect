import wandb
import argparse

import train
from cfgs.sweep_cfgs import sweep_cfgs


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-name", type=str, default="extreme_lr_sweep")
    parser.add_argument("--count", type=int, default=10)
    return parser.parse_args()


def main():
    args = arg_parser()
    sweep_id = wandb.sweep(sweep_cfgs[args.sweep_name], project=train.WANDB_PROJECT, entity=train.WANDB_ENTITY)
    wandb.agent(sweep_id, function=lambda: train.main(is_sweep=True), count=args.count)


if __name__ == "__main__":
    main()
