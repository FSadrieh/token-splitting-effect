#!/bin/bash
python src/evaluation/create_report.py imdb -t
python src/evaluation/evaluate_1_models.py imdb -t
python src/evaluation/evaluate_1_models.py short_imdb -t
python scripts/imdb/run_10_random_masking.py