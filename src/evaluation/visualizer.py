import argparse
import torch
import os
import random

import time
import pickle
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

from matplotlib import pyplot as plt
import matplotlib
from math import sqrt

from utils import create_soft_prompts, create_init_text, get_model_names_from_numbers, load_init_text

DEFAULT_COLORS = ["gray", "red", "blue", "orange", "green", "purple", "brown", "pink", "olive", "cyan", "black"]
DEFAULT_COLORS_LIGHT = ["#A6CEE4", "#B1DF8A", "#FB9B98", "#FEC06E", "#C9B2D6", "#FFFF99"]
DEFAULT_COLORS_STRONG = ["#2077B3", "#32A02E", "#E21A1C", "#FF7F00", "#6B3D9A", "#B15A29"]
SPINE_COLOR = 'gray'


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

def generate_random_float(small=5, high=10):
    if random.choice([True, False]):
        return random.uniform(-high, -small)
    else:
        return random.uniform(small, high)


def get_model_embedding_spaces(models: list) -> (torch.Tensor, list):
    from transformers.models.auto.modeling_auto import AutoModelForMaskedLM

    embeddings, labels = [], []
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
        mapper = umap.UMAP(metric="cosine", n_neighbors=30, n_jobs=-1).fit(reduced_embedding_space)
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
    colors: list = DEFAULT_COLORS_STRONG,
    labels: list = None,
) -> None:
    """
    Plots the embedding space in steps. First the embedding spaces from the different models are plotted with a low alpha value. Then the prompts and init texts are plotted with a higher alpha value.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    unique_labels = sorted(set(labels), reverse=True)
    
    label_to_color = {label: colors[i] for i, label in enumerate(sorted(set(labels)))}
    label_to_color_light = {label: DEFAULT_COLORS_LIGHT[i] for i, label in enumerate(unique_labels)}
    label_to_color_strong = {label: DEFAULT_COLORS_STRONG[i] for i, label in enumerate(unique_labels)}

    # Plot each point with its corresponding color
    for label in set(labels):
        indices = [i for i, l in enumerate(labels) if l == label]
        if label in prompts:
            ax.scatter(
                embedding_space[indices, 0],
                embedding_space[indices, 1],
                alpha=0.5,
                s=5,
                color=label_to_color_strong[label],
                label=label
            )
            for (i, (x, y)) in enumerate(zip(embedding_space[indices, 0], embedding_space[indices, 1])):
                ax.annotate(i, (x, y), color=label_to_color_strong[label])
        else:
            ax.scatter(
                embedding_space[indices, 0],
                embedding_space[indices, 1],
                alpha=0.01,
                s=5,
                color=label_to_color_light[label],
                label=label
            )
    legend = ax.legend()
    for lh in legend.legend_handles: 
        lh.set_alpha(1)
    fig, ax = plt.subplots(figsize=(10,8), dpi=300)
    unique_labels = sorted(set(labels))

    label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

    # Plot each point with its corresponding color and alpha
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        alpha = 0.5 if label in prompts else 0.008
        ax.scatter(
            embedding_space[indices, 0],
            embedding_space[indices, 1],
            alpha=alpha,
            s=5,
            color=label_to_color[label],
            label=label
        )
        if label in prompts:
            for i, (x, y) in enumerate(zip(embedding_space[indices, 0], embedding_space[indices, 1])):
                ax.annotate(f"{i}", (x, y), fontsize=16, weight="bold",  xytext=(x+generate_random_float(8, 15), y+generate_random_float(8, 15)), arrowprops=dict(arrowstyle="->", connectionstyle=f"arc3, rad={generate_random_float(0.1, 0.4)}", facecolor=label_to_color[label], color=label_to_color[label]), color=label_to_color[label])

    # Improving legend visibility
    legend = ax.legend(loc='best', fontsize='small', fancybox=True)
    for lh in legend.legend_handles: 
        lh.set_alpha(1)

    fig.savefig(os.path.join("visualizations", output_name))
    
def latexify(fig_width=None, fig_height=None, columns=2):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches
        #fig_width = 12 if columns==1 else 17 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              # 'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'lines.linewidth': 0.5,
              'axes.linewidth': 0.0,
              #'text.fontsize': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'lines.markersize': 2,
              #'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


def prepare_soft_prompts(
    soft_prompt_names: list, prompt_length: int, embedding_size: int, is_avg: bool
) -> (torch.Tensor, list, int):
    soft_prompt_list, labels = create_soft_prompts(soft_prompt_names, prompt_length, embedding_size)
    if is_avg:
        soft_prompt_list = [torch.mean(soft_prompt, dim=0) for soft_prompt in soft_prompt_list]
        prompt_length = 1
        labels = [label + "_avg" for label in labels]
    soft_prompts = torch.cat(soft_prompt_list, dim=0)
    return soft_prompts, prompt_length, labels


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
    method: str
):
    print(
        f"Visualizing soft prompts: {soft_prompt_names} and init text: {init_text}, on the models {model_names}. Pre averaging is {is_avg}. Method is {method}. Saving to {output_name}."
    )
    latexify()
    # soft_prompts, prompt_length, soft_prompt_labels = prepare_soft_prompts(soft_prompt_names, prompt_length, embedding_size, is_avg)
    # init_model = init_text_model if init_text_model else model_names[0]
    # if init_text:
    #     init_text, init_text_name = (
    #         load_init_text(soft_prompt_names[0])
    #         if init_text == "default"
    #         else create_init_text(init_text, init_model, prompt_length, embedding_size)
    #     )
    #     soft_prompts = torch.cat([soft_prompts, init_text], dim=0)
    #     soft_prompt_names.append(init_text_name)

    # embeddings, model_names, model_labels = get_model_embedding_spaces(model_names)
    # embedding_space = torch.cat([embeddings, soft_prompts], dim=0).detach().numpy()

    # reduced_embedding_space = reduce_embedding_space(embedding_space, method=method)
    # with open('cache.pickle', 'wb') as handle:
    #     pickle.dump([reduced_embedding_space, embeddings, model_names, prompt_length, soft_prompt_names, soft_prompt_labels, model_labels], handle)
    
    with open('cache.pickle', 'rb') as handle:
        reduced_embedding_space, embeddings, model_names, prompt_length, soft_prompt_names, soft_prompt_labels, model_labels = pickle.load(handle)

    plot_embedding_space(
        reduced_embedding_space,
        output_name=output_name,
        model_embedding_size=embeddings.shape[0] // len(model_names),
        prompt_length=prompt_length,
        prompts=soft_prompt_names,
        model_names=model_names,
        labels=model_labels + soft_prompt_labels,
    )


if __name__ == "__main__":
    main()
