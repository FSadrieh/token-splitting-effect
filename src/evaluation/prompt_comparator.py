import argparse
import torch
import csv

from utils import create_init_text, create_soft_prompts, get_model_names_from_numbers


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_names", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("-i", "--init_text", type=str, default=None)
    # Note it does not make a lot of sense to use the init text with soft prompts trained on other models, since the init text would be different
    parser.add_argument("-m", "--model_numbers", type=str, default=None)
    parser.add_argument("-pa", "--pre_averaging", type=bool, default=False)
    parser.add_argument(
        "-d", "--distance_metric", type=str, default="euclidean", help="Supports: euclidean_sim, euclidean, cosine, all"
    )
    parser.add_argument("-e", "--embedding_size", type=int, default=768)
    parser.add_argument("-p", "--prompt_length", type=int, default=16)

    return parser.parse_args()


def calculate_sim(soft_prompt_1: torch.Tensor, soft_prompt_2: torch.Tensor, distance_metric: str, pre_averaging: bool) -> float:
    if pre_averaging:
        soft_prompt_1 = torch.mean(soft_prompt_1, dim=1)
        soft_prompt_2 = torch.mean(soft_prompt_2, dim=1)

    if distance_metric == "cosine":
        sim = torch.nn.functional.cosine_similarity(soft_prompt_1, soft_prompt_2, dim=-1)
    elif distance_metric in ["euclidean", "euclidean_sim"]:
        distance = torch.nn.functional.pairwise_distance(soft_prompt_1, soft_prompt_2, p=2)
        if distance_metric == "euclidean_sim":
            if not pre_averaging:
                max_distance = torch.max(distance)
                sim = 1 - (distance / max_distance)
            else:
                sim = 1 - distance
        else:
            sim = distance

    else:
        raise ValueError("Invalid distance metric")

    if not pre_averaging:
        return torch.mean(sim).item()

    return sim.item()


def main():
    args = arg_parser()
    soft_prompt_names = args.soft_prompt_names.split(",")
    tokenizer_names = (
        get_model_names_from_numbers(args.model_numbers.split(",")) if args.model_numbers is not None else None
    )
    compare(
        soft_prompt_names,
        tokenizer_names,
        args.init_text,
        args.pre_averaging,
        args.distance_metric,
        args.embedding_size,
        args.prompt_length,
        args.output_path,
    )


def compare(
    soft_prompt_names: list,
    tokenizer_names: list,
    init_text: str,
    pre_averaging: bool,
    distance_metric: str,
    embedding_size: int,
    prompt_length: int,
    output_path: str,
):
    print(
        f"Comparing soft prompts: {soft_prompt_names} and init text: {init_text}, with the distance metric {distance_metric}, on the models {tokenizer_names}. Pre averaging is {pre_averaging}"
    )
    soft_prompt_list = create_soft_prompts(soft_prompt_names, prompt_length, embedding_size)

    # Creates the initial soft prompt if specified
    if init_text is not None:
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
        for j in range(i + 1, len(soft_prompt_list)):
            # If the distance metric is all, it calculates the similarity for all distance metrics and pre-averaging options
            if distance_metric == "all":
                sim_e_s = calculate_sim(soft_prompt_list[i], soft_prompt_list[j], "euclidean_sim", pre_averaging)
                sim_e_d = calculate_sim(soft_prompt_list[i], soft_prompt_list[j], "euclidean", pre_averaging)
                sim_c_s = calculate_sim(soft_prompt_list[i], soft_prompt_list[j], "cosine", pre_averaging)
                sim = f"\nEuclidean similarity: {sim_e_s}\nEuclidean distance: {sim_e_d}\nCosine similarity: {sim_c_s}"
            else:
                sim = calculate_sim(soft_prompt_list[i], soft_prompt_list[j], distance_metric, pre_averaging)
            similarity_dict[(soft_prompt_names[i], soft_prompt_names[j])] = round(sim, 3)
            similarity_dict[(soft_prompt_names[j], soft_prompt_names[i])] = round(sim, 3)

    for i in range(len(soft_prompt_names)):
        similarity_dict[(soft_prompt_names[i], soft_prompt_names[i])] = 0.0 if distance_metric == "euclidean" else 1.0

    with open(f"similarities/{output_path}", "w+") as f:
        writer = csv.writer(f)
        writer.writerow([""] + soft_prompt_names)
        for i in range(len(soft_prompt_names)):
            writer.writerow([soft_prompt_names[i]] + [similarity_dict[(soft_prompt_names[i], soft_prompt_names[j])] for j in range(len(soft_prompt_names))])


if __name__ == "__main__":
    main()
