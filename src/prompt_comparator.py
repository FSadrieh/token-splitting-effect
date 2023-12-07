import argparse
import torch

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s1", "--soft_prompt_path_1", type=str, required=True)
    parser.add_argument("-s2", "--soft_prompt_path_2", type=str, required=True)
    parser.add_argument("-pa", "--pre_averaging", type=bool, default=True)
    parser.add_argument("-d", "--distance_metric", type=str, default="euclidean")
    parser.add_argument("-e", "--embedding_size", type=str, default=768)
    parser.add_argument("-p", "--prompt_length", type=str, default=30)

    return parser.parse_args()

def create_soft_prompt(prompt_length: int, embedding_size: int, soft_prompt_path: str) -> torch.Tensor:
    soft_prompt = torch.nn.Embedding(prompt_length, embedding_size)
    soft_prompt.load_state_dict(torch.load(soft_prompt_path))
    prompt_tokens = torch.arange(prompt_length).long()
    return soft_prompt(prompt_tokens)

def calculate_sim(soft_prompt_1: torch.Tensor, soft_prompt_2: torch.Tensor, distance_metric: str, pre_averaging: bool) -> float:
    if pre_averaging:
        soft_prompt_1 = torch.mean(soft_prompt_1, dim=1)
        soft_prompt_2 = torch.mean(soft_prompt_2, dim=1)

    if distance_metric == "cosine":
        sim = torch.nn.functional.cosine_similarity(soft_prompt_1, soft_prompt_2, dim=-1)
    elif distance_metric == "euclidean":
        sim = 1- torch.nn.functional.pairwise_distance(soft_prompt_1, soft_prompt_2, p=2)
    else:
        raise ValueError("Invalid distance metric")

    if not pre_averaging:
        return torch.mean(sim)

    return sim

def main():
    args = arg_parser()
    soft_prompt_1 = create_soft_prompt(args.prompt_length, args.embedding_size, args.soft_prompt_path_1)
    soft_prompt_2 = create_soft_prompt(args.prompt_length, args.embedding_size, args.soft_prompt_path_2)
    sim = calculate_sim(soft_prompt_1, soft_prompt_2, args.distance_metric, args.pre_averaging)
    print(sim)

if __name__ == "__main__":
    main()