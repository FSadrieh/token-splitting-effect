#!/bin/bash

# These are the 5_models with different pre-trained MultiBerts
python train.py --config cfgs/emotion/5_models/models/emotion_5_model_0.yml
python train.py --config cfgs/emotion/5_models/models/emotion_5_model_1.yml
python train.py --config cfgs/emotion/5_models/models/emotion_5_model_2.yml

# These are the 5_models with different soft prompt initialization
python train.py --config cfgs/emotion/5_models/init_seeds/emotion_5_model_init_1.yml
python train.py --config cfgs/emotion/5_models/init_seeds/emotion_5_model_init_2.yml

# These are the 5_models with different training seeds
python train.py --config cfgs/emotion/5_models/train_seeds/emotion_5_model_train_1.yml
python train.py --config cfgs/emotion/5_models/train_seeds/emotion_5_model_train_2.yml
