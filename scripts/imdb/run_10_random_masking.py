import yaml
import numpy as np

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/src" + "/evaluation")

from prompt_shortener import validate_prompt  # noqa: E402
from utils import get_model_numbers_from_names  # noqa: E402

# This script is to reproduce the results of the random masking experiment of the paper

SOFT_PROMPTS = ["10_model_0", "10_model_1", "10_model_2", "10_model_3", "10_model_4"]
CONFIG_PATH = "cfgs/imdb/10_models/"
SEEDS = [5653572532, 8943168, 794113451]


def main():
    losses = []
    for soft_prompt_name in SOFT_PROMPTS:
        config_path = f"{CONFIG_PATH}{soft_prompt_name}.yml"
        config = yaml.safe_load(open(config_path))
        model_numbers = get_model_numbers_from_names(config["hf_model_names"])
        for seed in SEEDS:
            for model_number in model_numbers:
                losses.append(
                    validate_prompt(
                        model_number=model_number,
                        config=config_path,
                        soft_prompt_name=soft_prompt_name,
                        accelerator="cuda",
                        prompt_length=config["prompt_length"],
                        embedding_size=768,
                        dropped_out_tokens=[],
                        inverse=False,
                        shorten=False,
                        batch_size=128,
                        use_test_set=True,
                        seed=seed,
                        number_of_tokens_to_drop=14,
                    )
                )

    print(losses)
    print(np.mean(losses))
    print(np.std(losses))


if __name__ == "__main__":
    main()
