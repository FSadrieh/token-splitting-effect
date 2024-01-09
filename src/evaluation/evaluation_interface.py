import argparse

from utils import get_model_names_from_numbers


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", metavar="command", required=True)

    parser.add_argument("soft_prompt_names", type=str)
    parser.add_argument("model_numbers", type=str)
    parser.add_argument("-e", "--embedding_size", type=int, default=768)
    parser.add_argument("-p", "--prompt_length", type=int, default=16)

    vis_parser = subparsers.add_parser("visualize", help="Visualize soft prompt embeddings.")
    vis_parser.add_argument("output_path", type=str)
    vis_parser.add_argument("-i", "--init_texts", type=str, default=None)
    vis_parser.add_argument(
        "-m",
        "--init_text_models",
        type=str,
        default=None,
        help="Comma separated list of models to use for init text. If not specified the models for the embeddings will be used",
    )
    vis_parser.add_argument("-a", "--average", action="store_true", default=False)

    comp_parser = subparsers.add_parser("compare", help="Compare soft prompt embeddings.")
    comp_parser.add_argument("-i", "--init_text", type=str, default=None)
    comp_parser.add_argument("-pa", "--pre_averaging", type=bool, default=False)
    comp_parser.add_argument(
        "-d", "--distance_metric", type=str, default="euclidean", help="Supports: euclidean_sim, euclidean, cosine, all"
    )

    back_parser = subparsers.add_parser("back_translation", help="Create back translation prompt.")
    back_parser.add_argument("-d", "--distance_metric", type=str, default="euclidean", help="Supports: euclidean, cosine")

    return parser.parse_args()


def main():
    args = parse_args()

    soft_prompt_names = args.soft_prompt_names.split(",")
    model_numbers = args.model_numbers.split(",")
    model_names = get_model_names_from_numbers(model_numbers)

    if args.command == "visualize":
        import visualizer as visualizer

        visualizer.visualize(
            soft_prompt_names,
            model_names,
            args.output_path,
            args.init_texts,
            args.init_text_models,
            args.average,
            args.embedding_size,
            args.prompt_length,
        )
    elif args.command == "compare":
        import prompt_comparator as prompt_comparator

        prompt_comparator.compare(
            soft_prompt_names,
            model_names,
            args.init_text,
            args.pre_averaging,
            args.distance_metric,
            args.embedding_size,
            args.prompt_length,
        )
    elif args.command == "back_translation":
        import back_translation as back_translation

        assert len(soft_prompt_names) == 1 and len(model_names) == 1
        back_translation.back_translation(
            soft_prompt_names[0], model_names[0], args.distance_metric, args.embedding_size, args.prompt_length
        )
    else:
        raise ValueError("Invalid command")


if __name__ == "__main__":
    main()
