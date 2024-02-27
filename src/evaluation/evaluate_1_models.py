import argparse
import numpy as np
import csv

from utils import validate_soft_prompt, create_trainer_etc


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_name",
        type=str,
        choices=["imdb", "emotion", "mnli", "short_imdb"],
        help="We have trained all MultiBERT models on IMDb, Emotion, and MNLI. We also have a short IMDb experiment, to evaluate token masking. ",
    )
    parser.add_argument("-t", "--use_test_set", action="store_true")
    parser.add_argument("-a", "--accelerator", type=str, default="cuda", help="Supports: cuda, cpu, tpu, mps")
    parser.add_argument("-e", "--embedding_size", type=int, default=768)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    return parser.parse_args()


def main():
    """
    This script evaluates the loss of all 25 MultiBERT models for a given experiment. For the paper we have done this for the datasets IMDb, Emotion, and MNLI. In addition, we have a short IMDb experiment, to evaluate token masking.
    """
    args = arg_parser()
    if args.experiment_name == "short_imdb":
        prompt_length = 2
        config_dir = "cfgs/imdb/1_models_short_prompt/"
        soft_prompt_names = [f"1_model_{i}_sp" for i in range(25)]
    elif args.experiment_name == "imdb":
        prompt_length = 16
        config_dir = "cfgs/imdb/1_models/"
        soft_prompt_names = [f"1_model_{i}" for i in range(25)]
    else:
        prompt_length = 16
        config_dir = f"cfgs/{args.experiment_name}/1_models/"
        soft_prompt_names = [f"{args.experiment_name}_1_model_{i}" for i in range(25)]
    config = config_dir + soft_prompt_names[0] + ".yml"

    one_model_loss = []
    model_args, trainer, dm = create_trainer_etc(
        config, "google/multiberts-seed_0", args.accelerator, prompt_length, args.batch_size
    )

    for i, soft_prompt_name in enumerate(soft_prompt_names):
        one_model_loss.append(
            validate_soft_prompt(
                model_args=model_args,
                trainer=trainer,
                dm=dm,
                model_number=i,
                use_test_set=args.use_test_set,
                soft_prompt_name=soft_prompt_name,
            )
        )

    with open(f"{config_dir}/one_losses.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["model_number", "val_loss"])
        for i, loss in enumerate(one_model_loss):
            writer.writerow([i, loss])
        writer.writerow(["mean", np.mean(one_model_loss)])
        writer.writerow(["std", np.std(one_model_loss)])


if __name__ == "__main__":
    main()
