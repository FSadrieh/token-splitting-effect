#!/bin/bash
python train.py --config cfgs/mnli/10_models/mnli_10_model_0.yml
python train.py --config cfgs/mnli/10_models/mnli_10_model_1.yml
python train.py --config cfgs/mnli/10_models/mnli_10_model_2.yml