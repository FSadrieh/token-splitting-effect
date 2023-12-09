import argparse
import torch

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_names", type=str)
    parser.add_argument("-pa", "--pre_averaging", type=bool, default=True)
    parser.add_argument("-d", "--distance_metric", type=str, default="euclidean")
    parser.add_argument("-e", "--embedding_size", type=str, default=768)
    parser.add_argument("-p", "--prompt_length", type=str, default=30)

    return parser.parse_args()

def create_soft_prompt(prompt_length: int, embedding_size: int, soft_prompt_name: str) -> torch.Tensor:
    soft_prompt_path = f"logs/explainable-soft-prompts/{soft_prompt_name}/checkpoints/soft_prompt.pt"
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
        distance = torch.nn.functional.pairwise_distance(soft_prompt_1, soft_prompt_2, p=2)
        if not pre_averaging: #TODO: Check if this is correct
            max_distance = torch.max(distance)
            sim = 1 - (distance / max_distance)
        else:
            sim = 1 - distance
    else:
        raise ValueError("Invalid distance metric")

    if not pre_averaging:
        return torch.mean(sim)

    return sim

def main():
    args = arg_parser()
    soft_prompt_names = args.soft_prompt_names.split(",")
    if len(soft_prompt_names) < 2:
        raise ValueError("You need to specify at least two soft prompts")
    soft_prompt_list = []
    for soft_prompt_name in soft_prompt_names:
        soft_prompt_list.append(create_soft_prompt(args.prompt_length, args.embedding_size, soft_prompt_name))
    for i in range(len(soft_prompt_list)):
        for j in range(i+1, len(soft_prompt_list)):
            sim = calculate_sim(soft_prompt_list[i], soft_prompt_list[j], args.distance_metric, args.pre_averaging)
            print(soft_prompt_names[i], soft_prompt_names[j], sim)

if __name__ == "__main__":
    main()