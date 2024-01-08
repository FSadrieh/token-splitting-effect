import argparse
import torch
from transformers.models.auto.modeling_auto import AutoModelForMaskedLM
from transformers import AutoTokenizer

from utils import create_soft_prompt


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_name", type=str)
    parser.add_argument("model_number", type=str)
    parser.add_argument(
        "-d", "--distance_metric", type=str, default="euclidean", help="Supports: euclidean, cosine"
    )
    parser.add_argument("-e", "--embedding_size", type=str, default=768)
    parser.add_argument("-p", "--prompt_length", type=str, default=16)

    return parser.parse_args()

def main():
    args = arg_parser()
    soft_prompt = create_soft_prompt(args.soft_prompt_name, args.prompt_length, args.embedding_size)

    model_name = f"google/multiberts-seed_{args.model_number}"
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    embeddings = model.get_input_embeddings().weight

    tokens = []
    for i in range(soft_prompt.shape[0]):
        soft_prompt_token = soft_prompt[i].unsqueeze(0)
        if args.distance_metric == "cosine":
            distance = torch.nn.functional.cosine_similarity(soft_prompt_token, embeddings, dim=-1) * - 1
        else:
            distance = torch.linalg.vector_norm(embeddings - soft_prompt_token, dim=-1, ord=2)
        tokens.append(torch.argmin(distance).item())

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print(tokenizer.decode(torch.tensor(tokens)))

if __name__ == "__main__":
    main()
