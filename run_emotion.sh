#!/bin/bash
python dlib/data/data_download.py --dataset emotion -o data

python train.py --config cfgs/emotion/2_models/emotion_2_model_0.yml
python train.py --config cfgs/emotion/2_models/emotion_2_model_1.yml
python train.py --config cfgs/emotion/2_models/emotion_2_model_2.yml

python train.py --config cfgs/emotion/5_models/models/emotion_5_model_0.yml
python train.py --config cfgs/emotion/5_models/models/emotion_5_model_1.yml
python train.py --config cfgs/emotion/5_models/models/emotion_5_model_2.yml

# These are the 5_models with different soft prompt initialization
python train.py --config cfgs/emotion/5_models/init_seeds/emotion_5_model_init_1.yml
python train.py --config cfgs/emotion/5_models/init_seeds/emotion_5_model_init_2.yml

# These are the 5_models with different training seeds
python train.py --config cfgs/emotion/5_models/train_seeds/emotion_5_model_train_1.yml
python train.py --config cfgs/emotion/5_models/train_seeds/emotion_5_model_train_2.yml

#10 models
python train.py --config cfgs/emotion/10_models/emotion_10_model_0.yml
python train.py --config cfgs/emotion/10_models/emotion_10_model_1.yml
python train.py --config cfgs/emotion/10_models/emotion_10_model_2.yml

# 1 models
python train.py --config cfgs/emotion/1_models/emotion_1_model_0.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_1.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_2.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_3.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_4.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_5.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_6.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_7.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_8.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_9.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_10.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_11.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_12.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_13.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_14.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_15.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_16.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_17.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_18.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_19.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_20.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_21.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_22.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_23.yml
python train.py --config cfgs/emotion/1_models/emotion_1_model_24.yml

# EVALUATION
python src/evaluation/create_report.py emotion -t
python src/evaluation/evaluate_1_models.py emotion -t