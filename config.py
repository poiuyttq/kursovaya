import copy
import json
from pathlib import Path


DEFAULT_CONFIG = {
    "map": "henon",
    "compute_method": "vector",
    "draw_trajectory": True,
    "draw_regime_map": True,
    "output_dir": "output",
    "initial_state": {
        "x": 0.0,
        "y": 0.0
    },
    "trajectory": {
        "steps": 100000,
        "skip_points": 100
    },
    "regime_map": {
        "x_min": -0.6,
        "x_max": 0.6,
        "y_min": 0.0,
        "y_max": 2.0,
        "x_steps": 220,
        "y_steps": 220,
        "transient_steps": 500,
        "sample_steps": 300,
        "divergence_check_steps": 1200,
        "max_period": 50,
        "divergence_limit": 10.0,
        "tolerance": 0.0001,
        "workers": 4,
        "show_progress": True
    },
    "parameters": {
        "henon": {
            "b": 0.2,
            "alfa": 1.4
        },
        "logistic": {
            "r": 3.6
        },
        "power_abs": {
            "mu": 1.0,
            "A": 0.0,
            "nu": 0.8,
            "C": 1.3
        }
    }
}


def deep_update(base, override):
    result = copy.deepcopy(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value

    return result


def load_config(path):
    config_path = Path(path)

    if not config_path.exists():
        config_path.write_text(
            json.dumps(DEFAULT_CONFIG, indent=2),
            encoding="utf-8",
        )
        print(f"Config file was not found. Created default '{config_path}'.")

    with config_path.open("r", encoding="utf-8") as file:
        user_config = json.load(file)

    config = deep_update(DEFAULT_CONFIG, user_config)
    validate_config(config)

    return config


def save_config(config, path):
    config_path = Path(path)
    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def validate_config(config):
    available_maps = {"henon", "logistic", "power_abs"}
    available_methods = {"vector", "parallel"}

    if config["map"] not in available_maps:
        raise ValueError(f"Unknown map '{config['map']}'. Use one of {available_maps}.")

    if config["compute_method"] not in available_methods:
        raise ValueError(
            f"Unknown compute_method '{config['compute_method']}'. "
            f"Use one of {available_methods}."
        )

    if config["regime_map"]["max_period"] < 1:
        raise ValueError("regime_map.max_period must be >= 1.")
