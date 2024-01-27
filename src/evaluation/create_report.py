import argparse
import yaml
import numpy as np
import csv

from token_relevance_evaluation import Token_Relevance_Evaluation
from utils import get_model_numbers_from_names

DEFAULT_EMBEDDING_SIZE = 768

DEFAULT_CONFIG_PATHS = {
    "emotion": "cfgs/emotion/",
    "imdb": "cfgs/imdb/",
    }

EXAMINED_SOFT_PROMPTS = {
    "emotion":
        {
            "emotion_2_model": [3, "2_models/"],
            "emotion_5_model_init": [3, "5_models/init_seeds/"],	
            "emotion_5_model": [3, "5_models/models/"],
            "emotion_5_model_train": [3, "5_models/train_seeds/"],
            "emotion_10_model": [3, "10_models/"],
        },
    "imdb":
        {
            "2_model": [5, "2_models/"],
            "5_model_init": [5, "5_models/init_seeds/"],
            "5_model": [5, "5_models/models/"],
            "5_model_train": [5, "5_models/train_seeds/"],
            "10_model": [5, "10_models/"],
        },
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["emotion", "imdb"])
    parser.add_argument("-k", "--k", type=int, default=7)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-a", "--accelerator", type=str, default="cuda", help="Supports: cuda, cpu, tpu, mps")
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        with open(
            f"logs/explainable-soft-prompts/{args.dataset}.csv",
            "r",
        ) as f:
            print(f"File {args.dataset}.csv already exists. Please delete or rename it first.")
            return
    except FileNotFoundError:
        pass

    euc_accuracy = []
    cos_accuracy = []
    only_good_tokens_loss = []
    shortend_prompt_loss = []
    avg_model_losses = []
    
    for soft_prompt_class, [num_soft_prompts, config_path] in EXAMINED_SOFT_PROMPTS[args.dataset].items():
        class_euc_accuracy = []
        class_cos_accuracy = []
        class_normal_length_tokens_loss = []
        class_shortend_prompt_loss = []
        class_model_loss = []
        for i in range(num_soft_prompts):
            soft_prompt_name = f"{soft_prompt_class}_{i}"
            config_path = DEFAULT_CONFIG_PATHS[args.dataset] + config_path + f"{soft_prompt_class}_{i}.yml"
            config = yaml.safe_load(open(config_path))
            evaluator = Token_Relevance_Evaluation(soft_prompt_name, get_model_numbers_from_names(config["hf_model_names"]), config_path, args.accelerator, config["prompt_length"], DEFAULT_EMBEDDING_SIZE, args.k, args.batch_size)
            evaluator.token_relevance_evaluation()
            class_euc_accuracy.append(evaluator.euc_accuracy)
            class_cos_accuracy.append(evaluator.cos_accuracy)
            class_normal_length_tokens_loss.append(np.mean(evaluator.normal_prompt_length_loss))
            class_shortend_prompt_loss.append(np.mean(evaluator.shortend_loss))
            class_model_loss.append(np.mean(evaluator.model_loss_per_token))
        
        euc_accuracy.append(np.mean(class_euc_accuracy))
        cos_accuracy.append(np.mean(class_cos_accuracy))
        only_good_tokens_loss.append(np.mean(class_normal_length_tokens_loss))
        shortend_prompt_loss.append(np.mean(class_shortend_prompt_loss))
        avg_model_losses.append(np.mean(class_model_loss))

    with open(
        f"logs/explainable-soft-prompts/{args.dataset}.csv",
        "w+",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metrics"] + EXAMINED_SOFT_PROMPTS[args.dataset].keys())
        writer.writerow(["euc_accuracy"] + euc_accuracy)
        writer.writerow(["cos_accuracy"] + cos_accuracy)
        writer.writerow(["only_good_tokens_loss"] + only_good_tokens_loss)
        writer.writerow(["shortend_prompt_loss"] + shortend_prompt_loss)
        writer.writerow(["avg_model_losses"] + avg_model_losses)


if __name__ == "__main__":
    main()