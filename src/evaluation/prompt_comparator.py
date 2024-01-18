import argparse
import csv
from os import path
import sys

from utils import create_init_text, create_soft_prompts, get_model_names_from_numbers, load_init_text

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from similarity_calculation import calculate_sim  # noqa: E402


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "soft_prompt_names",
        type=str,
        help="Comma separated list of soft prompt names. If you do not know what soft prompt names are available check logs/explainable-soft-prompts.",
    )
    parser.add_argument(
        "output_name", type=str, help="Path to save the plot to. The plot will be saved in the similarities folder."
    )
    parser.add_argument(
        "-i",
        "--init_text",
        type=str,
        default=None,
        help="If set the init text will be used in the similarity calculation. If init_text=='default' the init text from the specified soft prompt will be used.",
    )
    # Note it does not make a lot of sense to use the init text with soft prompts trained on other models, since the init text would be different
    parser.add_argument(
        "-m",
        "--model_numbers",
        type=str,
        default=None,
        help="Comma separated list of model numbers to use for init text. If not specified the models for the soft prompts will be used. Will be ignored if init_text is not set or set to 'default'.",
    )
    parser.add_argument(
        "-pa",
        "--pre_averaging",
        action="store_true",
        help="If set the similarity is calculated between the average of the prompt tokens of each prompt. Does not make sense to use with euclidean similarity.",
    )
    # TODO: Should we remove euclidean similarity?
    parser.add_argument(
        "-d", "--distance_metric", type=str, default="euclidean", help="Supports: euclidean_sim, euclidean, cosine"
    )
    parser.add_argument(
        "-sbpt",
        "--similarity_between_prompt_tokens",
        action="store_true",
        help="If set sim between all prompts is calculated. Otherwise, the similarity between each token of the prompts is calculated. If pre-average is set, this flag is ignored.",
    )
    parser.add_argument("-e", "--embedding_size", type=int, default=768)
    parser.add_argument("-p", "--prompt_length", type=int, default=16)

    return parser.parse_args()


def main():
    args = arg_parser()
    soft_prompt_names = args.soft_prompt_names.split(",")
    tokenizer_names = get_model_names_from_numbers(args.model_numbers.split(",")) if args.model_numbers is not None else None
    if args.distance_metric not in ["euclidean_sim", "euclidean", "cosine"]:
        raise ValueError("The distance metric must be euclidean_sim, euclidean or cosine")
    compare(
        soft_prompt_names,
        tokenizer_names,
        args.init_text,
        args.pre_averaging,
        args.distance_metric,
        args.embedding_size,
        args.prompt_length,
        args.output_name,
        args.similarity_between_prompt_tokens,
    )


def compare(
    soft_prompt_names: list,
    tokenizer_names: list,
    init_text: str,
    pre_averaging: bool,
    distance_metric: str,
    embedding_size: int,
    prompt_length: int,
    output_name: str,
    similarity_between_prompt_tokens: bool,
):
    print(
        f"Comparing soft prompts: {soft_prompt_names} and init text: {init_text}, with the distance metric {distance_metric}, on the models {tokenizer_names}. Pre averaging is {pre_averaging}."
    )
    soft_prompt_list = create_soft_prompts(soft_prompt_names, prompt_length, embedding_size)

    # Creates the initial soft prompt if specified
    if init_text is not None:
        if init_text == "default":
            for i in range(len(soft_prompt_names)):
                try:
                    init_prompt, init_name = load_init_text(soft_prompt_names[i])
                    soft_prompt_list.append(init_prompt)
                    soft_prompt_names.append(init_name)
                except FileNotFoundError:
                    print(f"Could not find the initial prompt for {soft_prompt_names[i]}")
        else:
            if tokenizer_names is None:
                raise ValueError("You need to specify a tokenizer if you want to use an init text")
            for tokenizer in tokenizer_names:
                soft_prompt_list.append(create_init_text(init_text, tokenizer, embedding_size, prompt_length))
                soft_prompt_names.append(f"init_{tokenizer.split('/')[-1]}")

    # If there is only one soft prompt, there is nothing to compare. (At this stage the init text is already added to the list)
    if len(soft_prompt_names) < 2:
        raise ValueError("You need to specify at least two soft prompts")

    # Calculates the similarity between each pair of soft prompts
    similarity_dict = {}
    for i in range(len(soft_prompt_list)):
        for j in range(i, len(soft_prompt_list)):
            # If the distance metric is all, it calculates the similarity for all distance metrics and pre-averaging options
            sim = calculate_sim(
                soft_prompt_list[i], soft_prompt_list[j], distance_metric, pre_averaging, similarity_between_prompt_tokens
            )
            similarity_dict[(soft_prompt_names[i], soft_prompt_names[j])] = round(sim, 3)
            similarity_dict[(soft_prompt_names[j], soft_prompt_names[i])] = round(sim, 3)

    print(f"Saving the results to similarities/{output_name}.csv")
    with open(f"similarities/{output_name}.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow([""] + soft_prompt_names)
        for i in range(len(soft_prompt_names)):
            writer.writerow(
                [soft_prompt_names[i]]
                + [similarity_dict[(soft_prompt_names[i], soft_prompt_names[j])] for j in range(len(soft_prompt_names))]
            )


if __name__ == "__main__":
    main()
