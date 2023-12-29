import torch


def create_init_text(init_text: str, model_name: str, embedding_size: int, prompt_length: int) -> torch.Tensor:
    """
    This function mirrors the code in src/model.py to get the init text, to see how much the soft prompt changed
    """
    from transformers import AutoTokenizer
    import math
    from transformers.models.auto.modeling_auto import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
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


def create_init_texts(init_texts: list, model_names: list, prompt_length: int, embedding_size: int) -> (list, list):
    init_text_list = []
    init_texts_names = []
    for init_text in init_texts:
        for model_name in model_names:
            init_text_list.append(create_init_text(init_text, model_name, embedding_size, prompt_length))
            init_texts_names.append(f"{' '.join(init_text.split(' ')[:3])}_{model_name}")
    return init_text_list, init_texts_names


def create_soft_prompt(soft_prompt_name: str, prompt_length: int, embedding_size: int) -> torch.Tensor:
    soft_prompt_path = f"logs/explainable-soft-prompts/{soft_prompt_name}/checkpoints/soft_prompt.pt"
    soft_prompt = torch.nn.Embedding(prompt_length, embedding_size)
    soft_prompt.load_state_dict(torch.load(soft_prompt_path))
    prompt_tokens = torch.arange(prompt_length).long()
    return soft_prompt(prompt_tokens)


def create_soft_prompts(soft_prompt_names: list, prompt_length: int, embedding_size: int) -> list:
    soft_prompt_list = []
    for soft_prompt_name in soft_prompt_names:
        soft_prompt_list.append(create_soft_prompt(soft_prompt_name, prompt_length, embedding_size))
    return soft_prompt_list


def average(soft_prompt_list: list, average: bool) -> (torch.Tensor, int):
    if average:
        return torch.cat([torch.mean(soft_prompt, dim=0) for soft_prompt in soft_prompt_list], dim=0), 1
    return torch.cat(soft_prompt_list, dim=0), soft_prompt_list[0].shape[0]
