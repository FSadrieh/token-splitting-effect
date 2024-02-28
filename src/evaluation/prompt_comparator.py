import argparse
import csv
from os import path
import sys

from utils import load_soft_prompt_weights, load_init_text

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
        "output_name", type=str, help="Path to save the csv to. The csv will be saved in the similarities folder."
    )
    parser.add_argument(
        "-i",
        "--use_init_text",
        action="store_true",
        help="If set the init text will be used in the similarity calculation.",
    )
    parser.add_argument(
        "-pa",
        "--pre_averaging",
        action="store_true",
        help="If set the similarity is calculated between the average of the prompt tokens of each prompt.",
    )
    parser.add_argument("-d", "--distance_metric", type=str, default="euclidean", help="Supports: euclidean, cosine")
    parser.add_argument(
        "-sbpt",
        "--similarity_between_prompt_tokens",
        action="store_true",
        help="If set sim between all prompts is calculated. Otherwise, the similarity between each token of the prompts is calculated. If pre-average is set, this flag is ignored.",
    )

    return parser.parse_args()


def main():
    args = arg_parser()
    soft_prompt_names = args.soft_prompt_names.split(",")
    if args.distance_metric not in ["euclidean", "cosine"]:
        raise ValueError("The distance metric must be euclidean or cosine")
    compare(
        soft_prompt_names,
        args.init_text,
        args.pre_averaging,
        args.distance_metric,
        args.output_name,
        args.similarity_between_prompt_tokens,
    )


def compare(
    soft_prompt_names: list,
    init_text: bool,
    pre_averaging: bool,
    distance_metric: str,
    output_name: str,
    similarity_between_prompt_tokens: bool,
):
    """
    compare the similarity between soft prompts. If init_text is set, the init text will also be used in the comparison. For the different similarity metrics see src/similarity_calculation.py
    """
    print(
        f"Comparing soft prompts: {soft_prompt_names} and init text: {init_text}, with the distance metric {distance_metric}. Pre averaging is {pre_averaging}."
    )
    soft_prompt_list, __ = load_soft_prompt_weights(soft_prompt_names)

    # Creates the initial soft prompt if specified
    if init_text is not None:
        for i in range(len(soft_prompt_names)):
            try:
                init_prompt, init_name = load_init_text(soft_prompt_names[i])
                soft_prompt_list.append(init_prompt)
                soft_prompt_names.append(init_name)
            except FileNotFoundError:
                print(f"Could not find the initial prompt for {soft_prompt_names[i]}")

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
