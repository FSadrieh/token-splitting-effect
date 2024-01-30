#!/bin/bash
python train.py --config cfgs/emotion/2_models/emotion_2_model_0.yml
python train.py --config cfgs/emotion/2_models/emotion_2_model_1.yml
python train.py --config cfgs/emotion/2_models/emotion_2_model_2.yml