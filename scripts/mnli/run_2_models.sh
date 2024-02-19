#!/bin/bash
python train.py --config cfgs/mnli/2_models/mnli_2_model_0.yml
python train.py --config cfgs/mnli/2_models/mnli_2_model_1.yml
python train.py --config cfgs/mnli/2_models/mnli_2_model_2.yml