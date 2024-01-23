import argparse
from sklearn.neighbors import NearestNeighbors

from prompt_token_drop_out import prompt_token_drop_out


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_name", type=str)
    parser.add_argument("model_numbers", type=str, help="Comma separated list of model numbers to test on.")
    parser.add_argument("config", type=str, help="path to the config file for the validation")
    parser.add_argument("-a", "--accelerator", type=str, default="cuda", help="Supports: cuda, cpu, tpu, mps")
    parser.add_argument("-p", "--prompt_length", type=int, default=16)
    parser.add_argument("-e", "--embedding_size", type=int, default=768)
    return parser.parse_args()


def main():
    args = arg_parser()
    model_per_token = prompt_token_drop_out(
        args.model_numbers, args.config, args.soft_prompt_name, args.accelerator, args.prompt_length, args.embedding_size
    )

    neighbors = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(model_per_token)
    __, indices = neighbors.kneighbors(model_per_token)


if __name__ == "__main__":
    main()
