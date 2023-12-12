import argparse
import torch

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

DEFAULT_COLORS = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("soft_prompt_names", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("-e", "--embedding_size", type=str, default=768)
    parser.add_argument("-p", "--prompt_length", type=str, default=30)
    return parser.parse_args()


def create_soft_prompt(prompt_length: int, embedding_size: int, soft_prompt_name: str) -> torch.Tensor:
    soft_prompt_path = f"logs/explainable-soft-prompts/{soft_prompt_name}/checkpoints/soft_prompt.pt"
    soft_prompt = torch.nn.Embedding(prompt_length, embedding_size)
    soft_prompt.load_state_dict(torch.load(soft_prompt_path))
    prompt_tokens = torch.arange(prompt_length).long()
    return soft_prompt(prompt_tokens)


def get_model_embedding_space(model):
    from transformers.models.auto.modeling_auto import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(model)
    return model.get_input_embeddings().weight


def reduce_embedding_space(embedding_space, n_components=50):
    pca = PCA(n_components=n_components)
    reduced_embedding_space = pca.fit_transform(embedding_space)

    tsne = TSNE(random_state=1, metric="cosine")
    return tsne.fit_transform(reduced_embedding_space)


def plot_embedding_space(
    embedding_space, output_path, embedding_space_size, prompt_length, prompts, model, colors=DEFAULT_COLORS
):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        embedding_space[:embedding_space_size, 0],
        embedding_space[:embedding_space_size, 1],
        alpha=0.1,
        color=colors[0],
        label=f"Embeddings from {model}",
    )
    lower_bound = embedding_space_size
    for i in range(len(prompts)):
        upper_bound = lower_bound + prompt_length
        ax.scatter(
            embedding_space[lower_bound:upper_bound, 0],
            embedding_space[lower_bound:upper_bound, 1],
            alpha=1,
            color=colors[i + 1],
            label=f"Soft prompt {prompts[i]}",
        )
        lower_bound = upper_bound
    ax.legend()
    fig.savefig(output_path)


def main():
    args = arg_parser()

    soft_prompt_names = args.soft_prompt_names.split(",")
    soft_prompt_list = []
    # Creates a soft prompt for each soft prompt name
    for soft_prompt_name in soft_prompt_names:
        soft_prompt_list.append(create_soft_prompt(args.prompt_length, args.embedding_size, soft_prompt_name))
    soft_prompts = torch.cat(soft_prompt_list, dim=0)

    embeddings = get_model_embedding_space(args.model)
    embedding_space = torch.cat([embeddings, soft_prompts], dim=0).detach().numpy()

    reduced_embedding_space = reduce_embedding_space(embedding_space)
    plot_embedding_space(
        reduced_embedding_space,
        output_path=args.output_path,
        embedding_space_size=embeddings.shape[0],
        prompt_length=args.prompt_length,
        prompts=soft_prompt_names,
        model=args.model,
    )


if __name__ == "__main__":
    main()
