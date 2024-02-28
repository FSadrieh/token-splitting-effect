from typing import List, Tuple
import torch
from simple_parsing import parse_known_args
from transformers import AutoTokenizer
from lightning import Trainer, seed_everything

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from src.training.data_loading import LMDataModule  # noqa: E402
from args import TrainingArgs  # noqa: E402
from src.training.model import BasicLM  # noqa: E402


def create_init_text(init_text: str, model_name: str, embedding_size: int, prompt_length: int) -> torch.Tensor:
    """
    This function mirrors the code in src/model.py to get the init text, to see how much the soft prompt changed. This is for legacy runs, where the init text was not saved. This should not be used.
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


def create_init_texts(
    init_texts: list, model_names: list, prompt_length: int, embedding_size: int
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    This is for legacy runs, where the init text was not saved. This should not be used.
    """
    init_texts_list = []
    init_texts_names = []
    for init_text in init_texts:
        for model_name in model_names:
            init_texts_list.append(create_init_text(init_text, model_name, embedding_size, prompt_length))
            init_texts_names.append(f"{' '.join(init_text.split(' ')[:3])}_{model_name}")
    return init_texts_list, init_texts_names


def load_init_text(soft_prompt_name: str) -> Tuple[torch.Tensor, str]:
    """
    Loads the saved initialization of the SP, from init_soft_prompt.pt. There it it saved as a torch.Tensor.
    """
    init_text_path = f"logs/explainable-soft-prompts/{soft_prompt_name}/checkpoints/init_soft_prompt.pt"
    # Since we save a tensor we do not have to call weight here
    return torch.load(init_text_path), f"init text of {soft_prompt_name}"


def load_soft_prompt_weight(soft_prompt_name: str) -> torch.Tensor:
    """
    This method extracts the soft prompt weight from the saved soft prompt state dict.
    """
    soft_prompt_path = f"logs/explainable-soft-prompts/{soft_prompt_name}/checkpoints/soft_prompt.pt"
    return torch.load(soft_prompt_path)["weight"]


def load_soft_prompt_weights(soft_prompt_names: List[str]) -> Tuple[List[torch.Tensor], List[str]]:
    """
    This function is an interface to load multiple soft prompt weights at once, with load_soft_prompt_weight.
    We add labels required for the DR.
    """
    soft_prompt_list = []
    labels = []
    for soft_prompt_name in soft_prompt_names:
        soft_prompt = load_soft_prompt_weight(soft_prompt_name)
        soft_prompt_list.append(soft_prompt)
        labels.extend([soft_prompt_name] * soft_prompt.size(0))
    return soft_prompt_list, labels


# The following function are used to convert between model names and model numbers. This is used to make the code more readable and reduce the input a user has to give.
def get_model_names_from_numbers(model_numbers: list) -> list:
    return [f"google/multiberts-seed_{model_number}" for model_number in model_numbers]


def get_model_numbers_from_names(model_names: list) -> list:
    return [int(model_name.split("_")[-1]) for model_name in model_names]


def get_model_embedding_spaces(models: list, label_type: str) -> List[Tuple[torch.Tensor, List[str]]]:
    """
    For visualization purposes we want to get the model embeddings and the labels for the DR.
    As labels we can use the model names or the model numbers.
    """
    from transformers.models.auto.modeling_auto import AutoModelForMaskedLM

    embeddings = []
    labels = []
    for model in models:
        model_instance = AutoModelForMaskedLM.from_pretrained(model)
        model_embeddings = model_instance.get_input_embeddings().weight
        embeddings.append(model_embeddings)
        if label_type == "model_name":
            labels.extend([model] * model_embeddings.size(0))
        elif label_type == "model_number":
            labels.extend([int(model.split("_")[-1])] * model_embeddings.size(0))
        else:
            raise ValueError(f"Unknown label type: {label_type}")
    return torch.cat(embeddings, dim=0), labels


def get_k_nearest_neighbors(distance_metric: str, token: torch.Tensor, possible_neighbors: torch.Tensor, k: int) -> List[int]:
    """
    This function returns the k nearest neighbors of a token in the possible_neighbors.
    """
    if distance_metric == "cosine":
        similarity = torch.nn.functional.cosine_similarity(token, possible_neighbors, dim=-1)
        return torch.argsort(similarity, descending=True)[:k].tolist()
    else:
        distance = torch.linalg.vector_norm(possible_neighbors - token, dim=-1, ord=2)
        return torch.argsort(distance, descending=False)[:k].tolist()


def get_k_nearest_neighbors_for_all_tokens(
    distance_metric: str, soft_prompt: torch.Tensor, embeddings: torch.Tensor, k: int
) -> List[List[int]]:
    """
    This function returns the k nearest neighbors in an embedding space for all tokens in the soft prompt.
    """
    if k == 1:
        return [get_k_nearest_neighbors(distance_metric, soft_prompt[i], embeddings, k)[0] for i in range(soft_prompt.size(0))]
    return [get_k_nearest_neighbors(distance_metric, soft_prompt[i], embeddings, k) for i in range(soft_prompt.size(0))]


def create_trainer_etc(
    config: str, model_for_tokenizer: str, accelerator: str, prompt_length: int, batch_size: int = None
) -> Tuple[dict, LMDataModule, Trainer]:
    """
    This function creates everything that we need for a validation and test loop.
    We return a model_args since we want to change the MultiBERT models to the correct model names, before initializing the model, but can reuse the rest of the arguments.
    The trainer arguments are copied from the training script. Even though many of them are not needed here.
    """
    model_for_tokenizer = "google/multiberts-seed_0" if "multiberts" in model_for_tokenizer else model_for_tokenizer
    seed_everything(workers=True, seed=42)
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
        prompt_length=prompt_length,
        init_text=training_args.init_text,
        init_embedding_models=training_args.init_embedding_models,
        init_embedding_mode=training_args.init_embedding_mode,
        init_seed=training_args.init_seed,
    )
    if batch_size:
        training_args.eval_micro_batch_size = batch_size
        training_args.batch_size = batch_size
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


def validate_soft_prompt(
    model_args: dict,
    trainer: Trainer,
    dm: LMDataModule,
    model_number: int,
    use_test_set: bool = False,
    weight: torch.Tensor = None,
    soft_prompt_name: torch.Tensor = None,
) -> float:
    """
    This function validates/tests a soft prompt on a model. It returns the loss.
    """
    seed_everything(workers=True, seed=42)
    model_name = get_model_names_from_numbers([model_number])[0]

    model_args["model_names_or_paths"] = [model_name]
    if soft_prompt_name:
        print(f"Validating {soft_prompt_name} on model {model_number}")
        model_args["local_soft_prompt"] = f"logs/explainable-soft-prompts/{soft_prompt_name}/checkpoints/soft_prompt.pt"
    model = BasicLM(**model_args)
    if weight is not None:
        print(f"Validating custom weight on model {model_number}")
        model.set_soft_prompt_weight(weight)

    if use_test_set:
        return trainer.test(model, dm)[0]["test/loss"]
    return trainer.validate(model, dm)[0]["val/loss"]


def validate_soft_prompt_on_multiple_models(
    model_numbers: list,
    config_path: str,
    accelerator: str,
    prompt_length: int,
    batch_size: int,
    use_test_set: bool = False,
    weight: torch.Tensor = None,
    soft_prompt_name: str = None,
) -> List[float]:
    """
    This function validates/tests a soft prompt on a number of models. It returns the losses as a List.
    """
    model_names = get_model_names_from_numbers(model_numbers)
    model_args, dm, trainer = create_trainer_etc(config_path, model_names[0], accelerator, prompt_length, batch_size)
    val_losses = []
    for model_number in model_numbers:
        val_losses.append(validate_soft_prompt(model_args, trainer, dm, model_number, use_test_set, weight, soft_prompt_name))
    return val_losses
