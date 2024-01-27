#!/bin/bash
# You only need to run one of the first models since they are all the same

# These are the 5 models with different pre-trained MultiBerts
python train.py --config cfgs/imdb/5_models/models/5_model_0.yml
python train.py --config cfgs/imdb/5_models/models/5_model_1.yml
python train.py --config cfgs/imdb/5_models/models/5_model_2.yml
python train.py --config cfgs/imdb/5_models/models/5_model_3.yml
python train.py --config cfgs/imdb/5_models/models/5_model_4.yml

# These are the 5 models with different soft prompt initialization
# python train.py --config cfgs/imdb/5_models/init_seeds/5_model_init_0.yml
python train.py --config cfgs/imdb/5_models/init_seeds/5_model_init_1.yml
python train.py --config cfgs/imdb/5_models/init_seeds/5_model_init_2.yml
python train.py --config cfgs/imdb/5_models/init_seeds/5_model_init_3.yml
python train.py --config cfgs/imdb/5_models/init_seeds/5_model_init_4.yml

# These are the 5 models with different training seeds
# python train.py --config cfgs/imdb/5_models/train_seeds/5_model_train_0.yml
python train.py --config cfgs/imdb/5_models/train_seeds/5_model_train_1.yml
python train.py --config cfgs/imdb/5_models/train_seeds/5_model_train_2.yml
python train.py --config cfgs/imdb/5_models/train_seeds/5_model_train_3.yml
python train.py --config cfgs/imdb/5_models/train_seeds/5_model_train_4.yml