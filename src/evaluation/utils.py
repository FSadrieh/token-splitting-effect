import torch
from simple_parsing import parse_known_args
from transformers import AutoTokenizer
from lightning import Trainer
from typing import List

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from src.training.data_loading import LMDataModule  # noqa: E402
from args import TrainingArgs  # noqa: E402


def create_init_text(init_text: str, model_name: str, embedding_size: int, prompt_length: int) -> torch.Tensor:
    """
    This function mirrors the code in src/model.py to get the init text, to see how much the soft prompt changed. This is for legacy runs, where the init text was not saved.
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
    init_texts_list = []
    init_texts_names = []
    for init_text in init_texts:
        for model_name in model_names:
            init_texts_list.append(create_init_text(init_text, model_name, embedding_size, prompt_length))
            init_texts_names.append(f"{' '.join(init_text.split(' ')[:3])}_{model_name}")
    return init_texts_list, init_texts_names


def load_init_text(soft_prompt_name: str) -> (torch.Tensor, str):
    init_text_path = f"logs/explainable-soft-prompts/{soft_prompt_name}/checkpoints/init_soft_prompt.pt"
    # Since we save a tensor we do not have to call weight here
    return torch.load(init_text_path), f"init text of {soft_prompt_name}"


def create_soft_prompt(soft_prompt_name: str) -> torch.Tensor:
    """
    This method extracts the soft prompt weight from the saved soft prompt state dict.
    """
    soft_prompt_path = f"logs/explainable-soft-prompts/{soft_prompt_name}/checkpoints/soft_prompt.pt"
    return torch.load(soft_prompt_path)["weight"]


def create_soft_prompts(soft_prompt_names: list) -> list:
    soft_prompt_list = []
    labels = []
    for soft_prompt_name in soft_prompt_names:
        soft_prompt = create_soft_prompt(soft_prompt_name)
        # print(f"Soft prompt shape: {soft_prompt.shape}")
        soft_prompt_list.append(soft_prompt)
        labels.extend([soft_prompt_name] * soft_prompt.size(0))
    return soft_prompt_list, labels


def get_model_names_from_numbers(model_numbers: list) -> list:
    return [f"google/multiberts-seed_{model_number}" for model_number in model_numbers]


def create_trainer_etc(config: str, model_for_tokenizer: str, accelerator: str, prompt_length: int):
    training_args, __ = parse_known_args(TrainingArgs, config_path=config)

    tokenizer = AutoTokenizer.from_pretrained(model_for_tokenizer, use_fast=True)

    model_args = dict(
        tokenizer=tokenizer,
        from_scratch=training_args.from_scratch,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        beta1=training_args.beta1,
        beta2=training_args.beta2,
        lr_schedule=training_args.lr_schedule,
        warmup_period=training_args.warmup_period,
        prompt_length=training_args.prompt_length,
        init_text=training_args.init_text,
        init_embedding_models=training_args.init_embedding_models,
        init_embedding_mode=training_args.init_embedding_mode,
        init_seed=training_args.init_seed,
    )

    dm = LMDataModule(training_args=training_args, tokenizer=tokenizer, prompt_length=prompt_length)

    trainer = Trainer(
        max_epochs=training_args.training_goal,
        devices=training_args.num_devices,
        accelerator=accelerator,
        strategy=training_args.distributed_strategy,
        deterministic=training_args.force_deterministic,
        precision=training_args.precision,
        gradient_clip_val=training_args.grad_clip,
        inference_mode=not training_args.compile,  # inference_mode for val/test and PyTorch 2.0 compiler don't like each other
    )

    return model_args, dm, trainer


def get_model_embedding_spaces(models: list) -> (torch.Tensor, list):
    from transformers.models.auto.modeling_auto import AutoModelForMaskedLM

    embeddings = []
    labels = []
    for model in models:
        model_instance = AutoModelForMaskedLM.from_pretrained(model)
        model_embeddings = model_instance.get_input_embeddings().weight
        embeddings.append(model_embeddings)
        labels.extend([model] * model_embeddings.size(0))
    return torch.cat(embeddings, dim=0), models, labels
