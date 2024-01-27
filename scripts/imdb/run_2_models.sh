#!/bin/bash
python train.py --config cfgs/imdb/2_models/2_model_0.yml
python train.py --config cfgs/imdb/2_models/2_model_1.yml
python train.py --config cfgs/imdb/2_models/2_model_2.yml
python train.py --config cfgs/imdb/2_models/2_model_3.yml
python train.py --config cfgs/imdb/2_models/2_model_4.yml