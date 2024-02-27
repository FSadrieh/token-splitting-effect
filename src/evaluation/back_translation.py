import argparse
import torch
from transformers.models.auto.modeling_auto import AutoModelForMaskedLM
from transformers import AutoTokenizer

from utils import load_soft_prompt_weight, get_model_names_from_numbers, load_init_text, get_k_nearest_neighbors_for_all_tokens


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "soft_prompt_name",
        type=str,
        help="If you do not know what soft prompt names are available check logs/explainable-soft-prompts.",
    )
    parser.add_argument(
        "model_numbers",
        type=str,
        help="Comma separated list of model numbers to visualise the embeddings of. Model numbers are the numbers in the MultiBERT model names. For example, google/multiberts-seed_1 => 1.",
    )
    parser.add_argument(
        "-i", "--initial_prompt", action="store_true", help="Back translate the initial prompt instead of the trained one."
    )
    parser.add_argument("-d", "--distance_metric", type=str, default="euclidean", help="Supports: euclidean, cosine")
    return parser.parse_args()


def main():
    args = arg_parser()
    model_names = get_model_names_from_numbers(args.model_numbers.split(","))
    back_translation(args.soft_prompt_name, model_names, args.distance_metric, args.initial_prompt)


def back_translation(
    soft_prompt_name: str,
    model_names: str,
    distance_metric: str,
    use_initial_prompt: bool,
):
    if use_initial_prompt:
        print(
            f"Back translation for the initial prompt of {soft_prompt_name} on {model_names}, with the distance metric {distance_metric}."
        )
        soft_prompt, __ = load_init_text(soft_prompt_name)
    else:
        print(f"Back translation for {soft_prompt_name} on {model_names}, with the distance metric {distance_metric}.")
        soft_prompt = load_soft_prompt_weight(soft_prompt_name)

    # Back translation is done for every model, since the embeddings are different for each model.
    for model_name in model_names:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        embeddings = model.get_input_embeddings().weight

        # We get the k=1 nearest neighbors (The nearest neighbor) for each token in the prompt.
        tokens = get_k_nearest_neighbors_for_all_tokens(distance_metric, soft_prompt, embeddings, k=1)

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        decoded_tokens = [tokenizer.decode(torch.tensor(token)) for token in tokens]

        print(",".join(decoded_tokens))


if __name__ == "__main__":
    main()
