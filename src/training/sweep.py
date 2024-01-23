import wandb
import argparse

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
import train # noqa: E402
from cfgs.sweep_cfgs import sweep_cfgs # noqa: E402


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--config", type=str, help="Path to the config file, where the default values are set.")
    parser.add_argument("--count", type=int, default=10)
    return parser.parse_args()


def main():
    args = arg_parser()
    sweep_id = wandb.sweep(sweep_cfgs[args.name], project=train.WANDB_PROJECT, entity=train.WANDB_ENTITY)
    wandb.agent(sweep_id, function=lambda: train.main(is_sweep=True, config_path=args.config), count=args.count)


if __name__ == "__main__":
    main()
