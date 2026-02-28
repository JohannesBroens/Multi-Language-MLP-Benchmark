"""Hyperparameter tuning via grid search on PyTorch CUDA.

Selection strategy: Pareto-optimal on (accuracy, training_time).
Among configs within `acc_tolerance` of the best accuracy, pick the
fastest. This avoids chasing marginal accuracy gains that cost 10x
training time.
"""
import itertools
import os
import re
import subprocess
import sys

import yaml

PYTHON = sys.executable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATTERN_ACC = re.compile(r"Test Accuracy:\s+([\d.]+)%")
PATTERN_LOSS = re.compile(r"Test Loss:\s+([\d.]+)")
PATTERN_TIME = re.compile(r"Train time:\s+([\d.]+)\s*s")


def run_tuning(config, model, cache_path):
    """Grid search over optimizer x scheduler x batch_size x learning_rate.

    Captures accuracy, loss, and training time per trial.  Selects the
    Pareto-optimal config: fastest among those within `acc_tolerance`
    of peak accuracy.
    """
    tuning = config.get("tuning", {})
    search = tuning.get("search_space", {})

    batch_sizes = search.get("batch_size", [4096])
    learning_rates = search.get("learning_rate", [0.02])
    optimizers = search.get("optimizer", ["sgd"])
    schedulers = search.get("scheduler", ["none"])
    acc_tolerance = tuning.get("acc_tolerance", 1.0)  # percent

    if model == "cnn":
        script = os.path.join(PROJECT_ROOT, "src/python/models/cnn/cnn_pytorch.py")
        dataset_flag = "mnist"
    else:
        script = os.path.join(PROJECT_ROOT, "src/python/models/mlp/mlp_pytorch.py")
        dataset_flag = "generated"

    device = "cuda"
    tune_epochs = tuning.get("epochs", 50)

    trials = []
    total = len(batch_sizes) * len(learning_rates) * len(optimizers) * len(schedulers)
    trial_num = 0

    for bs, lr, opt, sched in itertools.product(batch_sizes, learning_rates, optimizers, schedulers):
        trial_num += 1
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

        print(f"[{trial_num}/{total}] bs={bs}, lr={lr}, opt={opt}, sched={sched_str} ... ",
              end="", flush=True)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            m_acc = PATTERN_ACC.search(result.stdout)
            m_loss = PATTERN_LOSS.search(result.stdout)
            m_time = PATTERN_TIME.search(result.stdout)

            if m_acc and m_time:
                acc = float(m_acc.group(1))
                loss = float(m_loss.group(1)) if m_loss else float("inf")
                train_time = float(m_time.group(1))
                print(f"acc={acc:.2f}%  loss={loss:.4f}  time={train_time:.3f}s")
                trials.append({
                    "batch_size": bs,
                    "learning_rate": lr,
                    "optimizer": opt,
                    "scheduler": sched_str,
                    "accuracy": acc,
                    "loss": loss,
                    "train_time": train_time,
                })
            else:
                print("FAILED (missing output fields)")
                if result.stderr:
                    print(f"  stderr: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
        except Exception as e:
            print(f"ERROR: {e}")

    if not trials:
        print("No successful trials.")
        return {}

    # --- Selection: Pareto-optimal (accuracy, time) ---
    best_acc = max(t["accuracy"] for t in trials)
    threshold = best_acc - acc_tolerance

    # Among configs within tolerance of best accuracy, pick fastest
    candidates = [t for t in trials if t["accuracy"] >= threshold]
    selected = min(candidates, key=lambda t: t["train_time"])

    best_params = {
        "batch_size": selected["batch_size"],
        "learning_rate": selected["learning_rate"],
        "optimizer": selected["optimizer"],
        "scheduler": selected["scheduler"],
    }

    # --- Save full results ---
    # Sort trials by accuracy descending for readability
    trials.sort(key=lambda t: (-t["accuracy"], t["train_time"]))

    output = {
        "selected": {
            **best_params,
            "accuracy": selected["accuracy"],
            "loss": selected["loss"],
            "train_time": selected["train_time"],
        },
        "selection_strategy": f"fastest within {acc_tolerance}% of peak accuracy ({best_acc:.2f}%)",
        "peak_accuracy": best_acc,
        "acc_tolerance": acc_tolerance,
        "tune_epochs": tune_epochs,
        "num_trials": len(trials),
        "trials": trials,
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"Peak accuracy:  {best_acc:.2f}%")
    print(f"Selected:       acc={selected['accuracy']:.2f}%  "
          f"loss={selected['loss']:.4f}  time={selected['train_time']:.3f}s")
    print(f"  bs={best_params['batch_size']}, lr={best_params['learning_rate']}, "
          f"opt={best_params['optimizer']}, sched={best_params['scheduler']}")
    print(f"Strategy:       fastest within {acc_tolerance}% of peak")
    print(f"Saved to {cache_path}")
    return best_params
