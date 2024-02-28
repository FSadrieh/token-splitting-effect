#!/bin/bash
python src/evaluation/create_report.py mnli -t
python src/evaluation/evaluate_1_models.py mnli -t