import torch


def calculate_sim(
    soft_prompt_1: torch.Tensor,
    soft_prompt_2: torch.Tensor,
    distance_metric: str,
    pre_averaging: bool,
    similarity_between_prompt_tokens: bool = False,
) -> float:
    """
    We have prompts p and p* with prompt tokens p_1, p_2, ..., p_n and p*_1, p*_2, ..., p*_n.
    There are three ways to calculate the similarity between p and p*:
        1. Calculate the similarity between the average of the prompt tokens of p and p* (pre_averaging=True). This means the prompts are first averaged along their token dimension and then the similarity is calculated. Since the similarity will be a single value, there is no need to average the similarity after the calculation.
        2. Calculate the similarity between the prompt tokens of p_i and p*_i (pre_averaging=False and similarity_between_prompt_tokens=False). This means the similarity between each prompt token of p and its "sibling" in p* is calculated and then averaged. sim = 1/n * sum_i(sim(p_i, p*_i))
        3. Calculate the similarity between each prompt token of p and p* (pre_averaging=False and similarity_between_prompt_tokens=True). This means the similarity between each prompt token of p and p* is calculated and then averaged. sim = 1/n^2 * sum_i(sum_j(sim(p_i, p*_j)))

        1. and 3. are presented in the Paper: "SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer" https://aclanthology.org/2022.acl-long.346/

    There are 2 distance metrics:
        1. Cosine similarity
        2. Euclidean distance
    """

    n = soft_prompt_1.shape[0]

    # This averaging reduces the size from (n, embedding_size) to (embedding_size)
    if pre_averaging:
        soft_prompt_1 = torch.mean(soft_prompt_1, dim=0)
        soft_prompt_2 = torch.mean(soft_prompt_2, dim=0)

    if distance_metric == "cosine":
        if pre_averaging or similarity_between_prompt_tokens:
            # This calculates the similarity of
            sim = torch.nn.functional.cosine_similarity(soft_prompt_1, soft_prompt_2, dim=-1)
            if not pre_averaging:
                return torch.mean(sim).item()
            return sim.item()

        sim = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sim[i, j] = torch.nn.functional.cosine_similarity(soft_prompt_1[i], soft_prompt_2[j], dim=-1)
        return torch.mean(sim).item()

    if pre_averaging or similarity_between_prompt_tokens:
        distance = torch.nn.functional.pairwise_distance(soft_prompt_1, soft_prompt_2, p=2)
    else:
        distance = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance[i, j] = torch.nn.functional.pairwise_distance(soft_prompt_1[i], soft_prompt_2[j], p=2)

    if not pre_averaging:
        return torch.mean(distance).item()

    return distance.item()
