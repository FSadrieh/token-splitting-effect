import argparse
import csv

from utils import load_init_text, validate_soft_prompt_on_multiple_models, get_model_names_from_numbers


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_name", type=str, help="The name of the soft prompt to validate.")
    parser.add_argument("model_numbers", type=str, help="Comma separated list of the model numbers the model was trained on.")
    parser.add_argument("config", type=str, help="path to the config file for the validation")
    parser.add_argument("-a", "--accelerator", type=str, default="cuda", help="Supports: cuda, cpu, tpu, mps")
    parser.add_argument("-p", "--prompt_length", type=int, default=16)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument(
        "-i", "--use_initial_prompt", action="store_true", help="If set the initial and trained version of the prompt are used."
    )
    parser.add_argument("-t", "--use_test_set", action="store_true")
    return parser.parse_args()


def main():
    """
    This function evaluates the loss of all 25 MultiBERT models for a given soft prompt. If use_initial_prompt is set, the prompt will be evaluated in the initial and trained stage.
    """
    args = arg_parser()
    checkpoint_name = f"logs/explainable-soft-prompts/{args.soft_prompt_name}/checkpoints/"

    val_losses = validate_soft_prompt_on_multiple_models(
        model_numbers=range(25),
        config_path=args.config,
        accelerator=args.accelerator,
        prompt_length=args.prompt_length,
        batch_size=args.batch_size,
        use_test_set=args.use_test_set,
        soft_prompt_name=args.soft_prompt_name,
    )

    if args.use_initial_prompt:
        soft_prompt_weight, __ = load_init_text(args.soft_prompt_name)
        i_val_losses = validate_soft_prompt_on_multiple_models(
            model_numbers=range(25),
            config_path=args.config,
            accelerator=args.accelerator,
            prompt_length=args.prompt_length,
            batch_size=args.batch_size,
            use_test_set=args.use_test_set,
            weight=soft_prompt_weight,
        )

    save_path = f"{checkpoint_name}/val_losses.csv"
    model_names = get_model_names_from_numbers(range(25))
    if args.use_initial_prompt:
        save(save_path, val_losses, model_names, args.model_numbers.split(","), i_val_losses)
    else:
        save(save_path, val_losses, model_names, args.model_numbers.split(","))


def save(save_path, val_losses, model_names, trained_on_model_numbers, i_val_losses=None):
    with open(save_path, "w+") as f:
        writer = csv.writer(f)
        if i_val_losses:
            writer.writerow(["seed", "initial_val_loss", "val_loss", "trained_on"])
            for i, val_loss in enumerate(val_losses):
                writer.writerow([model_names[i], i_val_losses[i], val_loss, "Yes" if i in trained_on_model_numbers else "No"])
        else:
            writer.writerow(["seed", "val_loss", "trained_on"])
            for i, val_loss in enumerate(val_losses):
                writer.writerow([model_names[i], val_loss, "Yes" if i in trained_on_model_numbers else "No"])


if __name__ == "__main__":
    main()
