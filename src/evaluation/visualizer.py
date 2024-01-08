import argparse
import torch
import os

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from utils import create_soft_prompts, create_init_texts, average, get_model_names_from_numbers

DEFAULT_COLORS = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan", "black"]


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_names", type=str)
    parser.add_argument("model_numbers", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("-i", "--init_texts", type=str, default=None)
    parser.add_argument(
        "-m",
        "--init_text_models",
        type=str,
        default=None,
        help="Comma separated list of models to use for init text. If not specified the models for the embeddings will be used",
    )
    parser.add_argument("-a", "--average", action="store_true", default=False)
    parser.add_argument("-e", "--embedding_size", type=int, default=768)
    parser.add_argument("-p", "--prompt_length", type=int, default=16)
    return parser.parse_args()


def get_model_embedding_spaces(models: list) -> (torch.Tensor, list):
    from transformers.models.auto.modeling_auto import AutoModelForMaskedLM

    embeddings = []
    for model in models:
        model = AutoModelForMaskedLM.from_pretrained(model)
        embeddings.append(model.get_input_embeddings().weight)
    return torch.cat(embeddings, dim=0), models


def reduce_embedding_space(embedding_space: torch.Tensor, n_components: int = 50) -> torch.Tensor:
    pca = PCA(n_components=n_components)
    reduced_embedding_space = pca.fit_transform(embedding_space)

    tsne = TSNE(random_state=1, metric="cosine")
    return tsne.fit_transform(reduced_embedding_space)


def plot_embedding_space(
    embedding_space: torch.Tensor,
    output_path: str,
    embedding_space_size: int,
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
        upper_bound = lower_bound + embedding_space_size
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
    fig.savefig(os.path.join("visualizations", output_path))


def prepare_soft_prompts(
    soft_prompt_names: list, prompt_length: int, embedding_size: int, avg: bool
) -> (torch.Tensor, list, int):
    soft_prompt_list = create_soft_prompts(soft_prompt_names, prompt_length, embedding_size)
    soft_prompts, prompt_length = average(soft_prompt_list, avg)
    return soft_prompts, soft_prompt_names, prompt_length


def prepare_init_texts(
    init_texts: list,
    model_names: list,
    prompt_length: int,
    embedding_size: int,
    avg: bool,
    soft_prompts: torch.Tensor,
    soft_prompt_names: list,
) -> (torch.Tensor, list):
    if not init_texts:
        return soft_prompts, soft_prompt_names
    init_texts = init_texts.split(";")
    init_texts_list, init_text_names = create_init_texts(init_texts, model_names, prompt_length, embedding_size)
    init_texts, _ = average(init_texts_list, avg)
    soft_prompt_names.extend(init_text_names)
    return torch.cat([soft_prompts, init_texts], dim=0), soft_prompt_names


def main():
    args = arg_parser()
    soft_prompt_names = args.soft_prompt_names.split(",")
    model_names = get_model_names_from_numbers(args.model_numbers.split(","))
    visualize(
        soft_prompt_names,
        model_names,
        args.output_path,
        args.init_texts,
        args.init_text_models,
        args.average,
        args.embedding_size,
        args.prompt_length,
    )


def visualize(
    soft_prompt_names: list,
    model_names: list,
    output_path: str,
    init_texts: str,
    init_text_models: str,
    avg: bool,
    embedding_size: int,
    prompt_length: int,
):
    print(
        f"Visualizing soft prompts: {soft_prompt_names} and init text: {init_texts}, on the models {model_names}. Pre averaging is {avg}. Saving to {output_path}."
    )
    soft_prompts, soft_prompt_names, prompt_length = prepare_soft_prompts(
        soft_prompt_names.split(","), prompt_length, embedding_size, average
    )
    init_models = init_text_models.split(",") if init_text_models else model_names.split(",")
    soft_prompts, soft_prompt_names = prepare_init_texts(
        init_texts, init_models, prompt_length, embedding_size, average, soft_prompts, soft_prompt_names
    )

    embeddings, model_names = get_model_embedding_spaces(model_names.split(","))
    embedding_space = torch.cat([embeddings, soft_prompts], dim=0).detach().numpy()

    reduced_embedding_space = reduce_embedding_space(embedding_space)
    plot_embedding_space(
        reduced_embedding_space,
        output_path=output_path,
        embedding_space_size=embeddings.shape[0] // len(model_names),
        prompt_length=prompt_length,
        prompts=soft_prompt_names,
        model_names=model_names,
    )


if __name__ == "__main__":
    main()
