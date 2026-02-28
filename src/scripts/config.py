"""Layered YAML config loader.

Merge order: base.yaml <- models/{model}.yaml <- local.yaml <- CLI overrides.
"""
import os
import yaml
from copy import deepcopy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values win."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(model: str = "mlp", config_path: str = None, overrides: dict = None) -> dict:
    """Load layered config: base <- model <- local <- overrides."""
    # Base config
    base_path = config_path or os.path.join(CONFIGS_DIR, "base.yaml")
    with open(base_path) as f:
        config = yaml.safe_load(f)

    # Model-specific overrides
    model_path = os.path.join(CONFIGS_DIR, "models", f"{model}.yaml")
    if os.path.exists(model_path):
        with open(model_path) as f:
            model_config = yaml.safe_load(f) or {}
        config = deep_merge(config, model_config)

    # Local overrides (gitignored)
    local_path = os.path.join(CONFIGS_DIR, "local.yaml")
    if os.path.exists(local_path):
        with open(local_path) as f:
            local_config = yaml.safe_load(f) or {}
        config = deep_merge(config, local_config)

    # CLI overrides
    if overrides:
        config = deep_merge(config, overrides)

    return config
