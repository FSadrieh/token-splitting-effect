sweep_cfgs = {
    "base_sweep": {
        "method": "bayes",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 1e-5, "max": 1e-3},
            "weight_decay": {"min": 0.0, "max": 0.2},
            "beta1": {"min": 0.8, "max": 0.99},
            "beta2": {"min": 0.9, "max": 0.999},
            "warmup_period": {"min": 0.0, "max": 0.1},
            "batch_size": {"min": 128, "max": 512},
        },
    },
    "extreme_lr_sweep": {
        "method": "bayes",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 1e-3, "max": 1e1},
            "batch_size": {"min": 128, "max": 512},
        },
    },
    "other_dataset_sweeps": {
        "method": "bayes",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 1e-2, "max": 1e0},
            "batch_size": {"min": 128, "max": 512},
        },
    },
}
