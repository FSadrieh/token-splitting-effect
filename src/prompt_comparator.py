import argparse
import torch

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_names", type=str)
    parser.add_argument("-i", "--init_text", type=str, default=None)
    # Note it does not make a lot of sense to use the init text with soft prompts trained on other models, since the init text would be different
    parser.add_argument("-t", "--tokenizer", type=str, default=None)
    parser.add_argument("-pa", "--pre_averaging", type=bool, default=False)
    parser.add_argument("-d", "--distance_metric", type=str, default="cosine")
    parser.add_argument("-e", "--embedding_size", type=str, default=768)
    parser.add_argument("-p", "--prompt_length", type=str, default=30)

    return parser.parse_args()

def create_soft_prompt(prompt_length: int, embedding_size: int, soft_prompt_name: str) -> torch.Tensor:
    soft_prompt_path = f"logs/explainable-soft-prompts/{soft_prompt_name}/checkpoints/soft_prompt.pt"
    soft_prompt = torch.nn.Embedding(prompt_length, embedding_size)
    soft_prompt.load_state_dict(torch.load(soft_prompt_path))
    prompt_tokens = torch.arange(prompt_length).long()
    return soft_prompt(prompt_tokens)

def create_init_text(init_text: str, embedding_size: int, tokenizer: str, prompt_length: int) -> torch.Tensor:
    """
    This function mirrors the code in src/model.py to get the init text, to see how much the soft prompt changed
    """
    from transformers import AutoTokenizer
    import math
    from transformers.models.auto.modeling_auto import AutoModelForMaskedLM
    
    model = AutoModelForMaskedLM.from_pretrained(tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    soft_prompt = torch.nn.Embedding(prompt_length, embedding_size)
    
    init_ids = tokenizer(init_text)["input_ids"]
    if len(init_ids) > prompt_length:
        init_ids = init_ids[:prompt_length]
    elif len(init_ids) < prompt_length:
        num_reps = math.ceil(prompt_length / len(init_ids))
        init_ids = init_ids * num_reps
    init_ids = init_ids[:prompt_length]
    prompt_token_weights = model.get_input_embeddings()(torch.LongTensor(init_ids)).detach().clone()
    soft_prompt.weight = torch.nn.Parameter(prompt_token_weights.to(torch.float32))
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
        return torch.mean(sim).item()

    return sim.item()

def main():
    args = arg_parser()
    soft_prompt_names = args.soft_prompt_names.split(",")
    soft_prompt_list = []

    # Creates a soft prompt for each soft prompt name
    for soft_prompt_name in soft_prompt_names:
        soft_prompt_list.append(create_soft_prompt(args.prompt_length, args.embedding_size, soft_prompt_name))

    # Creates the initial soft prompt if specified
    if args.init_text is not None:
        if args.tokenizer is None:
            raise ValueError("You need to specify a tokenizer if you want to use an init text")
        soft_prompt_list.append(create_init_text(args.init_text, args.embedding_size, args.tokenizer, args.prompt_length))
        soft_prompt_names.append("init_text")

    # If there is only one soft prompt, there is nothing to compare. (At this stage the init text is already added to the list)
    if len(soft_prompt_names) < 2:
        raise ValueError("You need to specify at least two soft prompts")

    # Calculates the similarity between each pair of soft prompts
    for i in range(len(soft_prompt_list)):
        for j in range(i+1, len(soft_prompt_list)):
            # If the distance metric is all, it calculates the similarity for all distance metrics and pre-averaging options
            if args.distance_metric == "all":
                sim_e_t = calculate_sim(soft_prompt_list[i], soft_prompt_list[j], "euclidean", True)
                sim_e_f = calculate_sim(soft_prompt_list[i], soft_prompt_list[j], "euclidean", False)
                sim_c_t = calculate_sim(soft_prompt_list[i], soft_prompt_list[j], "cosine", True)
                sim_c_f = calculate_sim(soft_prompt_list[i], soft_prompt_list[j], "cosine", False)
                sim = [sim_e_t, sim_e_f, sim_c_t, sim_c_f]
            else:
                sim = calculate_sim(soft_prompt_list[i], soft_prompt_list[j], args.distance_metric, args.pre_averaging)
            print(soft_prompt_names[i], soft_prompt_names[j], sim)

if __name__ == "__main__":
    main()