"""Hyperparameter tuning via grid search on PyTorch CUDA."""
import itertools
import os
import re
import subprocess
import sys

import yaml

PYTHON = sys.executable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATTERN_ACC = re.compile(r"Test Accuracy:\s+([\d.]+)%")


def run_tuning(config, model, cache_path):
    """Grid search over optimizer x scheduler x batch_size x learning_rate."""
    tuning = config.get("tuning", {})
    search = tuning.get("search_space", {})

    batch_sizes = search.get("batch_size", [4096])
    learning_rates = search.get("learning_rate", [0.02])
    optimizers = search.get("optimizer", ["sgd"])
    schedulers = search.get("scheduler", ["none"])

    # Use PyTorch CUDA for speed (fall back to CPU)
    if model == "cnn":
        script = os.path.join(PROJECT_ROOT, "src/python/models/cnn/cnn_pytorch.py")
        dataset_flag = "mnist"
    else:
        script = os.path.join(PROJECT_ROOT, "src/python/models/mlp/mlp_pytorch.py")
        dataset_flag = "generated"

    device = "cuda"
    tune_epochs = tuning.get("epochs", 50)

    best_acc = -1.0
    best_params = {}
    total = len(batch_sizes) * len(learning_rates) * len(optimizers) * len(schedulers)
    trial = 0

    for bs, lr, opt, sched in itertools.product(batch_sizes, learning_rates, optimizers, schedulers):
        trial += 1
        sched_str = sched if sched else "none"
        cmd = [
            PYTHON, script,
            "--dataset", dataset_flag,
            "--device", device,
            "--batch-size", str(bs),
            "--learning-rate", str(lr),
            "--epochs", str(tune_epochs),
            "--optimizer", opt,
            "--scheduler", sched_str,
        ]

        print(f"[{trial}/{total}] bs={bs}, lr={lr}, opt={opt}, sched={sched_str} ... ", end="", flush=True)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            match = PATTERN_ACC.search(result.stdout)
            if match:
                acc = float(match.group(1))
                print(f"acc={acc:.2f}%")
                if acc > best_acc:
                    best_acc = acc
                    best_params = {
                        "batch_size": bs,
                        "learning_rate": lr,
                        "optimizer": opt,
                        "scheduler": sched_str,
                    }
            else:
                print("FAILED (no accuracy in output)")
                if result.stderr:
                    print(f"  stderr: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
        except Exception as e:
            print(f"ERROR: {e}")

    # Save results
    output = {
        "best_accuracy": best_acc,
        "best_params": best_params,
        "search_space": {
            "batch_sizes": batch_sizes,
            "learning_rates": learning_rates,
            "optimizers": optimizers,
            "schedulers": schedulers,
        },
        "tune_epochs": tune_epochs,
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False)

    print(f"\nBest: acc={best_acc:.2f}%, params={best_params}")
    print(f"Saved to {cache_path}")
    return best_params
