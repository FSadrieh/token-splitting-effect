#!/bin/bash
python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_0_e2.yml
python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_0_e10.yml
python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_0_e20.yml

python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_1_e2.yml
python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_1_e10.yml
python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_1_e20.yml

python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_2_e2.yml
python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_2_e10.yml
python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_2_e20.yml

