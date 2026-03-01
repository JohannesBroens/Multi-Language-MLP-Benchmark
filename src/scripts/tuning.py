"""Two-phase hyperparameter tuning with successive halving.

Phase 1: Throughput sweep — find optimal batch size by measuring samples/s.
Phase 2: LR sweep with successive halving — per optimizer track (SGD, Adam).
Phase 3: Scheduler comparison — cosine vs none for the winner.

Selection: highest throughput within acc_tolerance of peak accuracy.
"""
import os
import re
import subprocess
import sys

import yaml

PYTHON = sys.executable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATTERN_ACC = re.compile(r"Test Accuracy:\s+([\d.]+)%")
PATTERN_LOSS = re.compile(r"Test Loss:\s+([\d.]+)")
PATTERN_THROUGHPUT = re.compile(r"Throughput:\s+([\d.]+)\s*samples/s")


def _get_script(model):
    """Return (script_path, dataset_flag) for model."""
    if model == "cnn":
        return os.path.join(PROJECT_ROOT, "src/python/models/cnn/cnn_pytorch.py"), "mnist"
    return os.path.join(PROJECT_ROOT, "src/python/models/mlp/mlp_pytorch.py"), "generated"


def _run_trial(script, dataset, bs, lr, epochs, optimizer="sgd", scheduler="none"):
    """Run one PyTorch CUDA trial, return dict with accuracy/loss/throughput or None."""
    cmd = [
        PYTHON, script,
        "--dataset", dataset,
        "--device", "cuda",
        "--batch-size", str(bs),
        "--learning-rate", str(lr),
        "--epochs", str(epochs),
        "--optimizer", optimizer,
        "--scheduler", scheduler,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        m_acc = PATTERN_ACC.search(result.stdout)
        m_loss = PATTERN_LOSS.search(result.stdout)
        m_thr = PATTERN_THROUGHPUT.search(result.stdout)
        if m_acc and m_thr:
            return {
                "accuracy": float(m_acc.group(1)),
                "loss": float(m_loss.group(1)) if m_loss else float("inf"),
                "throughput": float(m_thr.group(1)),
            }
    except subprocess.TimeoutExpired:
        pass
    return None


def throughput_sweep(script, dataset, batch_sizes, lr, sweep_epochs):
    """Phase 1: find optimal batch size by throughput.

    Returns dict: {batch_size: throughput, ...}.
    """
    print("\n--- Phase 1: Throughput Sweep ---")
    results = {}
    for bs in batch_sizes:
        print(f"  bs={bs} ... ", end="", flush=True)
        trial = _run_trial(script, dataset, bs, lr, sweep_epochs)
        if trial:
            results[bs] = trial["throughput"]
            print(f"{trial['throughput']:.0f} samples/s")
        else:
            print("FAILED")
    return results


def lr_halving(script, dataset, batch_size, lr_candidates, optimizer, tune_epochs):
    """Phase 2: successive halving over LR candidates.

    Starts all candidates at tune_epochs/4, drops bottom half by accuracy,
    doubles epochs for survivors, repeats until 1-2 remain at full epochs.

    Returns list of dicts sorted by accuracy descending, each with:
        {lr, accuracy, loss, throughput, epochs_run}
    """
    print(f"\n--- Phase 2: {optimizer.upper()} LR Halving (bs={batch_size}) ---")
    survivors = list(lr_candidates)
    min_epochs = max(tune_epochs // 4, 5)
    current_epochs = min_epochs
    all_results = []

    while len(survivors) > 1 and current_epochs <= tune_epochs:
        round_results = []
        for lr in survivors:
            print(f"  lr={lr}, epochs={current_epochs} ... ", end="", flush=True)
            trial = _run_trial(script, dataset, batch_size, lr, current_epochs, optimizer)
            if trial:
                print(f"acc={trial['accuracy']:.2f}%")
                round_results.append({"lr": lr, "epochs_run": current_epochs, **trial})
            else:
                print("FAILED")

        if not round_results:
            break

        # Sort by accuracy, keep top half
        round_results.sort(key=lambda t: -t["accuracy"])
        all_results.extend(round_results)
        cutoff = max(len(round_results) // 2, 1)
        survivors = [r["lr"] for r in round_results[:cutoff]]
        print(f"  -> Survivors: {survivors}")

        current_epochs = min(current_epochs * 2, tune_epochs)

    # Final run for the last survivor(s) at full epochs if not already there
    for lr in survivors:
        already_run = any(r["lr"] == lr and r["epochs_run"] == tune_epochs for r in all_results)
        if not already_run and current_epochs != tune_epochs:
            print(f"  lr={lr}, epochs={tune_epochs} (final) ... ", end="", flush=True)
            trial = _run_trial(script, dataset, batch_size, lr, tune_epochs, optimizer)
            if trial:
                print(f"acc={trial['accuracy']:.2f}%")
                all_results.append({"lr": lr, "epochs_run": tune_epochs, **trial})
            else:
                print("FAILED")

    # Deduplicate: keep best (longest) run per LR
    best_per_lr = {}
    for r in all_results:
        lr = r["lr"]
        if lr not in best_per_lr or r["epochs_run"] > best_per_lr[lr]["epochs_run"]:
            best_per_lr[lr] = r
    results = sorted(best_per_lr.values(), key=lambda t: -t["accuracy"])
    return results


def _select_batch_size(throughputs, tolerance):
    """Pick largest batch size within tolerance of peak throughput."""
    peak = max(throughputs.values())
    threshold = peak * (1.0 - tolerance)
    candidates = [bs for bs, thr in throughputs.items() if thr >= threshold]
    return max(candidates)


def run_tuning(config, model, cache_path):
    """Two-phase tuning: throughput sweep -> LR halving -> scheduler comparison."""
    tuning = config.get("tuning", {})
    training = config.get("training", {})

    batch_sizes = tuning.get("batch_sizes", [4096])
    sgd_lrs = tuning.get("sgd_lr_range", [0.02])
    adam_lrs = tuning.get("adam_lr_range", [0.001])
    acc_tolerance = tuning.get("acc_tolerance", 1.0)
    thr_tolerance = tuning.get("throughput_tolerance", 0.05)
    tune_epochs = tuning.get("tune_epochs", 50)
    sweep_epochs = tuning.get("sweep_epochs", 5)
    default_lr = training.get("learning_rate", 0.02)

    script, dataset = _get_script(model)

    # --- Phase 1: Throughput sweep ---
    throughputs = throughput_sweep(script, dataset, batch_sizes, default_lr, sweep_epochs)
    if not throughputs:
        print("Phase 1 failed — no successful batch size trials.")
        return {}
    best_bs = _select_batch_size(throughputs, thr_tolerance)
    print(f"  => Selected batch_size={best_bs} "
          f"({throughputs[best_bs]:.0f} samples/s, "
          f"peak={max(throughputs.values()):.0f})")

    # --- Phase 2: LR halving (two tracks) ---
    sgd_results = lr_halving(script, dataset, best_bs, sgd_lrs, "sgd", tune_epochs)
    adam_results = lr_halving(script, dataset, best_bs, adam_lrs, "adam", tune_epochs)

    # Combine and select: highest throughput within acc_tolerance of peak
    all_candidates = []
    for r in sgd_results:
        all_candidates.append({**r, "optimizer": "sgd", "batch_size": best_bs})
    for r in adam_results:
        all_candidates.append({**r, "optimizer": "adam", "batch_size": best_bs})

    if not all_candidates:
        print("Phase 2 failed — no successful LR trials.")
        return {}

    best_acc = max(c["accuracy"] for c in all_candidates)
    threshold = best_acc - acc_tolerance
    viable = [c for c in all_candidates if c["accuracy"] >= threshold]
    selected = max(viable, key=lambda c: c["throughput"])

    # --- Phase 3: Scheduler comparison (optional) ---
    scheduler_cmp = {}
    print("\n--- Phase 3: Scheduler Comparison ---")
    for sched in ["none", "cosine"]:
        print(f"  scheduler={sched} ... ", end="", flush=True)
        trial = _run_trial(script, dataset, best_bs, selected["lr"],
                           tune_epochs, selected["optimizer"], sched)
        if trial:
            print(f"acc={trial['accuracy']:.2f}%  {trial['throughput']:.0f} samples/s")
            scheduler_cmp[sched] = trial
        else:
            print("FAILED")

    best_params = {
        "batch_size": best_bs,
        "learning_rate": selected["lr"],
        "optimizer": selected["optimizer"],
        "scheduler": "none",
    }

    # --- Save results ---
    output = {
        "selected": {
            **best_params,
            "accuracy": selected["accuracy"],
            "loss": selected["loss"],
            "throughput": selected["throughput"],
        },
        "phase1_throughputs": {str(k): v for k, v in sorted(throughputs.items())},
        "phase1_selected_bs": best_bs,
        "phase2_sgd": sgd_results,
        "phase2_adam": adam_results,
        "scheduler_comparison": scheduler_cmp,
        "peak_accuracy": best_acc,
        "acc_tolerance": acc_tolerance,
        "tune_epochs": tune_epochs,
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"Phase 1:  bs={best_bs} ({throughputs[best_bs]:.0f} samples/s)")
    print(f"Phase 2:  opt={selected['optimizer']}, lr={selected['lr']}, "
          f"acc={selected['accuracy']:.2f}%")
    if scheduler_cmp:
        for s, t in scheduler_cmp.items():
            print(f"  sched={s}: acc={t['accuracy']:.2f}%  {t['throughput']:.0f} samples/s")
    print(f"Selected: {best_params}")
    print(f"Saved to {cache_path}")
    return best_params
