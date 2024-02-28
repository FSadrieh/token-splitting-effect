#!/bin/bash
python dlib/data/data_download.py -o data --dataset mnli --max_train_size 10000

python train.py --config cfgs/mnli/2_models/mnli_2_model_0.yml
python train.py --config cfgs/mnli/2_models/mnli_2_model_1.yml
python train.py --config cfgs/mnli/2_models/mnli_2_model_2.yml

python train.py --config cfgs/mnli/5_models/models/mnli_5_model_0.yml
python train.py --config cfgs/mnli/5_models/models/mnli_5_model_1.yml
python train.py --config cfgs/mnli/5_models/models/mnli_5_model_2.yml

# These are the 5_models with different soft prompt initialization
python train.py --config cfgs/mnli/5_models/init_seeds/mnli_5_model_init_1.yml
python train.py --config cfgs/mnli/5_models/init_seeds/mnli_5_model_init_2.yml

# These are the 5_models with different training seeds
python train.py --config cfgs/mnli/5_models/train_seeds/mnli_5_model_train_1.yml
python train.py --config cfgs/mnli/5_models/train_seeds/mnli_5_model_train_2.yml

#10 models
python train.py --config cfgs/mnli/10_models/mnli_10_model_0.yml
python train.py --config cfgs/mnli/10_models/mnli_10_model_1.yml
python train.py --config cfgs/mnli/10_models/mnli_10_model_2.yml

# One models
python train.py --config cfgs/mnli/1_models/mnli_1_model_0.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_1.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_2.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_3.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_4.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_5.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_6.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_7.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_8.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_9.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_10.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_11.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_12.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_13.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_14.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_15.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_16.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_17.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_18.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_19.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_20.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_21.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_22.yml
python train.py --config cfgs/mnli/1_models/mnli_1_model_23.yml

# Different training length mnli
python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_0_e2.yml
python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_0_e10.yml
python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_0_e20.yml

python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_1_e2.yml
python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_1_e10.yml
python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_1_e20.yml

python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_2_e2.yml
python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_2_e10.yml
python train.py --config cfgs/mnli/5_models/trainings_length/mnli_5_model_2_e20.yml

#EVALUATION
python src/evaluation/create_report.py mnli -t
python src/evaluation/evaluate_1_models.py mnli -t