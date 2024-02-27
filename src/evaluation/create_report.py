import argparse
import yaml
import numpy as np
import csv
from typing import Dict

from token_relevance_evaluation import Token_Relevance_Evaluation
from utils import get_model_numbers_from_names

DEFAULT_EMBEDDING_SIZE = 768

DEFAULT_CONFIG_PATHS = {
    "emotion": "cfgs/emotion/",
    "imdb": "cfgs/imdb/",
    "mnli": "cfgs/mnli/",
}

EXAMINED_SOFT_PROMPTS = {
    "emotion": {
        "emotion_2_model": [3, "2_models/"],
        "emotion_5_model_init": [3, "5_models/init_seeds/"],
        "emotion_5_model": [3, "5_models/models/"],
        "emotion_5_model_train": [3, "5_models/train_seeds/"],
        "emotion_10_model": [3, "10_models/"],
    },
    "imdb": {
        "2_model": [5, "2_models/"],
        "5_model_init": [5, "5_models/init_seeds/"],
        "5_model": [5, "5_models/models/"],
        "5_model_train": [5, "5_models/train_seeds/"],
        "10_model": [5, "10_models/"],
    },
    "mnli": {
        "mnli_2_model": [3, "2_models/"],
        "mnli_5_model_init": [3, "5_models/init_seeds/"],
        "mnli_5_model_e2": [3, "5_models/trainings_length/"],
        "mnli_5_model": [3, "5_models/models/"],
        "mnli_5_model_e10": [3, "5_models/trainings_length/"],
        "mnli_5_model_e20": [3, "5_models/trainings_length/"],
        "mnli_5_model_train": [3, "5_models/train_seeds/"],
        "mnli_10_model": [3, "10_models/"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["emotion", "imdb", "mnli"])
    parser.add_argument("-k", "--k", type=int, default=7)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-a", "--accelerator", type=str, default="cuda", help="Supports: cuda, cpu, tpu, mps")
    parser.add_argument("-t", "--use_test_set", action="store_true")
    return parser.parse_args()


def read_csv(csv_name):
    with open(
        f"logs/explainable-soft-prompts/{csv_name}",
        "r",
    ) as __:
        print(f"File {csv_name} already exists. Please delete or rename it first.")
        return


def main():
    args = parse_args()
    csv_name = f"{args.dataset}_test.csv" if args.use_test_set else f"{args.dataset}.csv"
    try:
        read_csv(csv_name)
    except FileNotFoundError:
        metrics = create_report(args.dataset, args.k, args.batch_size, args.accelerator, args.use_test_set)
        save_report(csv_name, args.dataset, metrics)


def create_report(dataset: str, k: int, batch_size: int, accelerator: str, use_test_set: bool) -> Dict[str, list]:
    def _calculate_mean_std(data):
        return np.mean(data), np.std(data)

    metrics = {
        "euc_accuracy": [],  # Alignment between the prompt location and prompt performance with euclidean metric
        "euc_std": [],
        "cos_accuracy": [],  # Alignment between the prompt location and prompt performance with cosine metric
        "cos_std": [],
        "masked_loss": [],  # Loss when the unimportant tokens are masked
        "masked_loss_std": [],
        "compressed_loss": [],  # Loss when the unimportant tokens are removed
        "compressed_loss_std": [],
        "important_token_masking_loss": [],  # Loss of the important tokens in token masking
        "important_token_masking_loss_std": [],
        "individual_model_loss": [],  # Loss of the SPs on the individual models
        "individual_model_loss_std": [],
    }

    for soft_prompt_class, [num_soft_prompts, config_path] in EXAMINED_SOFT_PROMPTS[dataset].items():
        class_metrics = create_report_per_class(
            soft_prompt_class, num_soft_prompts, config_path, dataset, k, batch_size, accelerator, use_test_set
        )
        for metric_name, metric_values in class_metrics.items():
            mean, std = _calculate_mean_std(metric_values)
            metrics[metric_name].append(mean)
            metrics[f"{metric_name}_std"].append(std)
    return metrics


def create_report_per_class(
    soft_prompt_class: str,
    num_soft_prompts: int,
    config_path: str,
    dataset: str,
    k: int,
    batch_size: int,
    accelerator: str,
    use_test_set: bool,
) -> Dict[str, list]:
    class_metrics = {
        "euc_accuracy": [],
        "cos_accuracy": [],
        "masked_loss": [],
        "compressed_loss": [],
        "important_token_masking_loss": [],
        "individual_model_loss": [],
    }

    for i in range(num_soft_prompts):
        print(f"Running evaluation for {soft_prompt_class}_{i}")
        # Here we handle the special cases for each dataset
        if soft_prompt_class in ["mnli_5_model_e2", "mnli_5_model_e10", "mnli_5_model_e20"]:
            prefix = "mnli_5_model_"
            suffix = soft_prompt_class.split(prefix)[1]
            soft_prompt_name = f"{prefix}{i}_{suffix}"
        else:
            soft_prompt_name = f"{soft_prompt_class}_{i}"
        model_config_path = DEFAULT_CONFIG_PATHS[dataset] + config_path + f"{soft_prompt_name}.yml"
        if i == 0 and ("init" in soft_prompt_name or "train" in soft_prompt_name):
            soft_prompt_name = "5_model_0" if dataset == "imdb" else f"{dataset}_5_model_0"
            model_config_path = DEFAULT_CONFIG_PATHS[dataset] + "5_models/models/" + f"{soft_prompt_name}.yml"

        config = yaml.safe_load(open(model_config_path))
        sp_metrics = create_report_for_soft_prompt(
            soft_prompt_name, model_config_path, config, k, batch_size, accelerator, use_test_set
        )
        for metric_name, metric_values in sp_metrics.items():
            class_metrics[metric_name].append(metric_values)

    return class_metrics


def create_report_for_soft_prompt(
    soft_prompt_name: str, model_config_path: str, config: dict, k: int, batch_size: int, accelerator: str, use_test_set: bool
) -> Dict[str, list]:
    evaluator = Token_Relevance_Evaluation(
        soft_prompt_name,
        get_model_numbers_from_names(config["hf_model_names"]),
        model_config_path,
        accelerator,
        config["prompt_length"],
        DEFAULT_EMBEDDING_SIZE,
        k,
        batch_size,
        use_test_set,
    )
    evaluator.token_relevance_evaluation()
    return {
        "euc_accuracy": evaluator.euc_accuracy,
        "cos_accuracy": evaluator.cos_accuracy,
        "masked_loss": evaluator.masked_loss,
        "compressed_loss": evaluator.compressed_loss,
        "important_token_masking_loss": evaluator.model_loss_per_token,
        "individual_model_loss": evaluator.individual_model_loss,
    }


def save_report(csv_name: str, dataset, metrics: Dict[str, list]):
    with open(
        f"logs/explainable-soft-prompts/{csv_name}",
        "w+",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metrics"] + list(EXAMINED_SOFT_PROMPTS[dataset].keys()))
        for metric_name, metric_values in metrics.items():
            writer.writerow([metric_name] + metric_values)


if __name__ == "__main__":
    main()
