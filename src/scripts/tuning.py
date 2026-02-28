"""Hyperparameter tuning via grid search."""
import os

import yaml


def run_tuning(config, model, cache_path):
    """Grid search over batch_size x learning_rate."""
    # TODO: implement grid search
    # For now, write current config values as "tuned"
    tuning = config.get("tuning", {})
    training = config.get("training", {})
    result = {
        "best_batch_size": training.get("batch_size"),
        "best_learning_rate": training.get("learning_rate"),
        "metric": tuning.get("metric", "accuracy"),
        "note": "placeholder - grid search not yet implemented",
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        yaml.dump(result, f)
    print(f"Tuning results saved to {cache_path}")
