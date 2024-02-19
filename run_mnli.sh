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