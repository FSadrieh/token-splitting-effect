import argparse
from collections import Counter
from typing import List
import torch
import csv

from prompt_token_drop_out import prompt_token_drop_out
from prompt_shortener import validate_prompt
from utils import (
    get_model_embedding_spaces,
    get_model_names_from_numbers,
    load_soft_prompt_weight,
    get_k_nearest_neighbors_for_all_tokens,
)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_name", type=str)
    parser.add_argument("model_numbers", type=str, help="Comma separated list of model numbers to test on.")
    parser.add_argument("config", type=str, help="path to the config file for the validation")
    parser.add_argument("-a", "--accelerator", type=str, default="cuda", help="Supports: cuda, cpu, tpu, mps")
    parser.add_argument("-p", "--prompt_length", type=int, default=16)
    parser.add_argument("-e", "--embedding_size", type=int, default=768)
    parser.add_argument("-k", "--k", type=int, default=7)
    return parser.parse_args()


def main():
    args = arg_parser()
    model_per_token = prompt_token_drop_out(
        args.model_numbers, args.config, args.soft_prompt_name, args.accelerator, args.prompt_length, args.embedding_size
    )
    model_numbers = args.model_numbers.split(",")
    losses_for_best_prompt_tokens = []
    for i, model_number in enumerate(model_numbers):
        tokens_to_drop = [j for j, x in enumerate(model_per_token) if x == i]
        prompt_shorten_args = {
            "model_number": model_number,
            "config": args.config,
            "soft_prompt_name": args.soft_prompt_name,
            "accelerator": args.accelerator,
            "prompt_length": args.prompt_length,
            "embedding_size": args.embedding_size,
            "dropped_out_tokens": tokens_to_drop,
            "inverse": True,
            "shorten": False,
        }
        normal_prompt_length_loss = validate_prompt(**prompt_shorten_args)
        prompt_shorten_args["shorten"] = True
        shortend_loss = validate_prompt(**prompt_shorten_args)
        losses_for_best_prompt_tokens.append((normal_prompt_length_loss, shortend_loss))

    model_names = get_model_names_from_numbers(model_numbers)
    model_embedding_spaces, labels = get_model_embedding_spaces(model_names, label_type="model_number")
    soft_prompt_weight = load_soft_prompt_weight(args.soft_prompt_name)

    euc_nn_votes = nearest_neighbor_vote("euclidean", soft_prompt_weight, model_embedding_spaces, labels, args.k)
    cos_nn_votes = nearest_neighbor_vote("cosine", soft_prompt_weight, model_embedding_spaces, labels, args.k)

    # Check if neighbor vote model correspond to the model through loss TODO: Can we include uncertainty into this average?
    euc_waccuracy = sum((euc_nn_votes[i][0] == model_per_token[i]) * euc_nn_votes[i][1] for i in range(len(euc_nn_votes))) / sum(euc_nn_votes[i][1] for i in range(len(euc_nn_votes)))
    cos_waccuracy = sum((cos_nn_votes[i][0] == model_per_token[i]) * cos_nn_votes[i][1] for i in range(len(cos_nn_votes))) / sum(cos_nn_votes[i][1] for i in range(len(cos_nn_votes)))
    euc_to_cos_accuracy = sum(euc_nn_votes[i][0] == cos_nn_votes[i][0] for i in range(len(euc_nn_votes))) / len(euc_nn_votes)

    with open(
        f"logs/explainable-soft-prompts/{args.soft_prompt_name}/checkpoints/{model_numbers.replace(',','_')}_token_drop_out.csv",
        "w+",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["seed"] + [i for i in range(len(args.prompt_length))])
        writer.writerow(["Loss assignment"] + model_per_token)
        writer.writerow(["Euclidean assignment"] + [euc_nn_votes[i][0] for i in range(len(euc_nn_votes))])
        writer.writerow(["Cosine assignment"] + [cos_nn_votes[i][0] for i in range(len(cos_nn_votes))])
        writer.writerow(["Euclidean certanty"] + [euc_nn_votes[i][1] for i in range(len(euc_nn_votes))])
        writer.writerow(["Cosine certanty"] + [cos_nn_votes[i][1] for i in range(len(cos_nn_votes))])
        writer.writerow([""])
        writer.writerow(["Euclidean weighted accuracy"] + [euc_waccuracy])
        writer.writerow(["Cosine weighted accuracy"] + [cos_waccuracy])
        writer.writerow(["Euclidean to cosine accuracy"] + [euc_to_cos_accuracy])
        writer.writerow([""])
        writer.writerow(["Prompt compression"] + [model_name for model_name in model_names])
        writer.writerow(["Normal prompt length"] + [loss[0] for loss in losses_for_best_prompt_tokens])
        writer.writerow(["Shortened prompt length"] + [loss[1] for loss in losses_for_best_prompt_tokens])


def nearest_neighbor_vote(
    distance_metric: str, soft_prompt_weight: torch.Tensor, model_embedding_spaces: list, labels: list, k: int = 7
) -> List[int]:
    nearest_neighbors = get_k_nearest_neighbors_for_all_tokens(distance_metric, soft_prompt_weight, model_embedding_spaces, k)
    nn_votes = []
    for token_neighbors in nearest_neighbors:
        neighbor_labels = [labels[neighbor] for neighbor in token_neighbors]
        most_common = Counter(neighbor_labels).most_common()[0]
        nn_votes.append((most_common[0], most_common[1] / k))

    return nn_votes


if __name__ == "__main__":
    main()
