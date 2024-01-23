import argparse
import torch
from transformers.models.auto.modeling_auto import AutoModelForMaskedLM
from transformers import AutoTokenizer

from utils import create_soft_prompt, get_model_names_from_numbers, load_init_text


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "soft_prompt_name",
        type=str,
        help="If you do not know what soft prompt names are available check logs/explainable-soft-prompts.",
    )
    parser.add_argument("model_numbers", type=str, help="Comma separated list of model numbers to visualise the embeddings of.")
    parser.add_argument(
        "-i", "--initial_prompt", action="store_true", help="Back translate the initial prompt instead of the trained one."
    )
    parser.add_argument("-d", "--distance_metric", type=str, default="euclidean", help="Supports: euclidean, cosine")
    return parser.parse_args()


def main():
    args = arg_parser()
    model_names = get_model_names_from_numbers(args.model_numbers.split(","))
    back_translation(
        args.soft_prompt_name, model_names, args.distance_metric, args.initial_prompt
    )


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
        soft_prompt = create_soft_prompt(soft_prompt_name)

    for model_name in model_names:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        embeddings = model.get_input_embeddings().weight

        tokens = []
        for i in range(soft_prompt.shape[0]):
            if distance_metric == "cosine":
                similarity = torch.nn.functional.cosine_similarity(soft_prompt[i], embeddings, dim=-1)
                tokens.append(torch.argmax(similarity).item())
            else:
                distance = torch.linalg.vector_norm(embeddings - soft_prompt[i], dim=-1, ord=2)
                tokens.append(torch.argmin(distance).item())

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        detokenized_tokens = [tokenizer.decode(torch.tensor(token)) for token in tokens]

        print(','.join(detokenized_tokens))


if __name__ == "__main__":
    main()
