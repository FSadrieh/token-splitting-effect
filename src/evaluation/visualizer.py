import argparse
import torch
import os

import time
import pickle
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

from matplotlib import pyplot as plt

from utils import create_soft_prompts, create_init_text, get_model_names_from_numbers, load_init_text

DEFAULT_COLORS = ["gray", "red", "blue", "orange", "green", "purple", "brown", "pink", "olive", "cyan", "black"]


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "soft_prompt_names",
        type=str,
        help="Comma separated list of soft prompt names. If you do not know what soft prompt names are available check logs/explainable-soft-prompts.",
    )
    parser.add_argument("model_numbers", type=str, help="Comma separated list of model numbers to visualise the embeddings of.")
    parser.add_argument(
        "output_name", type=str, help="Path to save the plot to. The plot will be saved in the visualizations folder."
    )
    parser.add_argument(
        "-i",
        "--init_text",
        type=str,
        default=None,
        help="If set the init text will be used in the plot. If init_text=='default' the init text from the specified soft prompt will be used.",
    )
    parser.add_argument(
        "-m",
        "--init_text_model",
        type=str,
        default=None,
        help="Comma separated list of models to use for init text. If not specified the models for the embeddings will be used. Will be ignored if init_text is not set or set to 'default'.",
    )
    parser.add_argument(
        "-a",
        "--average",
        action="store_true",
        default=False,
        help="If set the average of the soft prompts will be used instead of the soft prompts themselves.",
    )
    parser.add_argument("-e", "--embedding_size", type=int, default=768)
    parser.add_argument("-p", "--prompt_length", type=int, default=16)
    parser.add_argument("--method", type=str, default="umap", help="Method to use for dimensionality reduction, choose between 'umap' and 'tsne'.")
    return parser.parse_args()


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


def reduce_embedding_space(embedding_space: torch.Tensor, n_components: int = 50, method: str = "umap") -> torch.Tensor:
    print(f"Reducing embedding space to {n_components} dimensions.")
    start = time.time()
    pca = PCA(n_components=n_components)
    reduced_embedding_space = pca.fit_transform(embedding_space)
    print(f"PCA took {time.time() - start} seconds.")
    
    if method == "umap":
        start = time.time()
        mapper = umap.UMAP(metric="cosine", n_neighbors=50).fit(reduced_embedding_space)
        result = mapper.transform(reduced_embedding_space)
        print(f"UMAP took {time.time() - start} seconds.")
    elif method == "tsne":    
        tsne = TSNE(random_state=1, metric="cosine")
        result = tsne.fit_transform(reduced_embedding_space)
    else:
        raise ValueError(f"Method {method} is not supported.")

    return result


def plot_embedding_space(
    embedding_space: torch.Tensor,
    output_name: str,
    model_embedding_size: int,
    prompt_length: int,
    prompts: list,
    model_names: list,
    colors: list = DEFAULT_COLORS,
) -> None:
    """
    Plots the embedding space in steps. First the embedding spaces from the different models are plotted with a low alpha value. Then the prompts and init texts are plotted with a higher alpha value.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    lower_bound = 0
    for i in range(len(model_names)):
        upper_bound = lower_bound + model_embedding_size
        ax.scatter(
            embedding_space[lower_bound:upper_bound, 0],
            embedding_space[lower_bound:upper_bound, 1],
            alpha=0.1,
            color=colors[i],
            label=f"Embeddings from {model_names[i]}",
        )
        lower_bound = upper_bound
    for i in range(len(prompts)):
        upper_bound = lower_bound + prompt_length
        ax.scatter(
            embedding_space[lower_bound:upper_bound, 0],
            embedding_space[lower_bound:upper_bound, 1],
            alpha=1,
            color=colors[i + len(model_names)],
            label=f"{prompts[i]}",
        )
        lower_bound = upper_bound
    ax.legend()
    fig.savefig(os.path.join("visualizations", output_name))


def prepare_soft_prompts(
    soft_prompt_names: list, prompt_length: int, embedding_size: int, is_avg: bool
) -> (torch.Tensor, list, int):
    soft_prompt_list = create_soft_prompts(soft_prompt_names, prompt_length, embedding_size)
    if is_avg:
        soft_prompt_list = [torch.mean(soft_prompt, dim=0) for soft_prompt in soft_prompt_list]
        prompt_length = 1
    soft_prompts = torch.cat(soft_prompt_list, dim=0)
    return soft_prompts, prompt_length


def main():
    args = arg_parser()
    soft_prompt_names = args.soft_prompt_names.split(",")
    model_names = get_model_names_from_numbers(args.model_numbers.split(","))
    visualize(
        soft_prompt_names,
        model_names,
        args.output_name,
        args.init_text,
        args.init_text_model,
        args.average,
        args.embedding_size,
        args.prompt_length,
        args.method
    )


def visualize(
    soft_prompt_names: list,
    model_names: list,
    output_name: str,
    init_text: str,
    init_text_model: str,
    is_avg: bool,
    embedding_size: int,
    prompt_length: int,
):
    print(
        f"Visualizing soft prompts: {soft_prompt_names} and init text: {init_text}, on the models {model_names}. Pre averaging is {is_avg}. Saving to {output_name}."
    )
    soft_prompts, prompt_length = prepare_soft_prompts(soft_prompt_names, prompt_length, embedding_size, is_avg)
    init_model = init_text_model if init_text_model else model_names[0]
    if init_text:
        init_text, init_text_name = (
            load_init_text(soft_prompt_names[0])
            if init_text == "default"
            else create_init_text(init_text, init_model, prompt_length, embedding_size)
        )
        soft_prompts = torch.cat([soft_prompts, init_text], dim=0)
        soft_prompt_names.append(init_text_name)

    embeddings, model_names = get_model_embedding_spaces(model_names)
    embedding_space = torch.cat([embeddings, soft_prompts], dim=0).detach().numpy()

    reduced_embedding_space = reduce_embedding_space(embedding_space)
    plot_embedding_space(
        reduced_embedding_space,
        output_name=output_name,
        model_embedding_size=embeddings.shape[0] // len(model_names),
        prompt_length=prompt_length,
        prompts=soft_prompt_names,
        model_names=model_names,
    )


if __name__ == "__main__":
    main()
