"""Three-phase hyperparameter tuning with successive halving.

Phase 1: Batch size sweep — double from min_batch_size until OOM or VRAM limit,
         pick largest within throughput_tolerance of peak.
Phase 2: LR sweep with successive halving — per optimizer track (SGD, Adam).
Phase 3: Scheduler comparison — cosine vs none for the Phase 2 winner.

Selection: highest throughput within acc_tolerance of peak accuracy.
"""
import os
import re
import subprocess
import sys
import threading

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


def _run_trial(script, dataset, bs, lr, epochs, optimizer="sgd", scheduler="none",
               num_samples=0, hidden_size=0):
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
    if num_samples > 0:
        cmd.extend(["--num-samples", str(num_samples)])
    if hidden_size > 0:
        cmd.extend(["--hidden-size", str(hidden_size)])
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


def _get_gpu_memory():
    """Query GPU memory via nvidia-smi. Returns (used_mb, total_mb) or None."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return float(parts[0].strip()), float(parts[1].strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


class _VRAMMonitor:
    """Poll peak VRAM usage in a background thread during a trial."""

    def __init__(self, poll_interval=0.3):
        self.poll_interval = poll_interval
        self.peak_used = 0.0
        self.total = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        while not self._stop.is_set():
            mem = _get_gpu_memory()
            if mem:
                self.peak_used = max(self.peak_used, mem[0])
                self.total = mem[1]
            self._stop.wait(self.poll_interval)

    def start(self):
        self.peak_used = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.peak_used, self.total


def throughput_sweep(script, dataset, min_bs, lr, sweep_epochs, vram_safety=0.90,
                     num_samples=0, hidden_size=0):
    """Phase 1: double batch size until OOM or VRAM safety limit.

    Starts at min_bs and keeps doubling.  Stops when:
      - The trial OOMs / crashes, or
      - Peak VRAM usage >= vram_safety fraction of total GPU memory.

    Returns dict: {batch_size: throughput, ...}.
    """
    print("\n--- Phase 1: Throughput Sweep (dynamic doubling) ---")
    results = {}
    monitor = _VRAMMonitor()
    bs = min_bs

    while True:
        # Ensure num_samples >= 2*bs so batching is meaningful
        effective_ns = max(num_samples, bs * 2) if num_samples > 0 else 0
        print(f"  bs={bs} ... ", end="", flush=True)
        monitor.start()
        trial = _run_trial(script, dataset, bs, lr, sweep_epochs,
                           num_samples=effective_ns, hidden_size=hidden_size)
        peak_used, total = monitor.stop()

        if not trial:
            print("FAILED (OOM)")
            break

        vram_pct = peak_used / total if total > 0 else 0
        results[bs] = trial["throughput"]
        print(f"{trial['throughput']:.0f} samples/s"
              f"  (VRAM: {peak_used:.0f}/{total:.0f} MB, {vram_pct:.0%})")

        if vram_pct >= vram_safety:
            print(f"  => VRAM {vram_pct:.0%} >= {vram_safety:.0%} safety limit, stopping")
            break

        bs *= 2

    return results


def lr_halving(script, dataset, batch_size, lr_candidates, optimizer, tune_epochs,
               num_samples=0, hidden_size=0):
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
            trial = _run_trial(script, dataset, batch_size, lr, current_epochs, optimizer,
                               num_samples=num_samples, hidden_size=hidden_size)
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
            trial = _run_trial(script, dataset, batch_size, lr, tune_epochs, optimizer,
                               num_samples=num_samples, hidden_size=hidden_size)
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

    min_bs = tuning.get("min_batch_size", 32)
    vram_safety = tuning.get("vram_safety", 0.90)
    sgd_lrs = tuning.get("sgd_lr_range", [0.02])
    adam_lrs = tuning.get("adam_lr_range", [0.001])
    acc_tolerance = tuning.get("acc_tolerance", 1.0)
    thr_tolerance = tuning.get("throughput_tolerance", 0.05)
    tune_epochs = tuning.get("tune_epochs", 50)
    sweep_epochs = tuning.get("sweep_epochs", 5)
    default_lr = training.get("learning_rate", 0.02)

    script, dataset = _get_script(model)

    # For MLP on "generated" dataset, use enough samples for batch_size to matter
    # CNN uses MNIST (60K samples) so num_samples is not needed
    scaling = config.get("scaling", {})
    hidden_size = training.get("hidden_size", 128)
    num_samples = scaling.get("fixed_num_samples", 0) if model != "cnn" else 0

    # --- Phase 1: Throughput sweep (double until OOM or VRAM limit) ---
    throughputs = throughput_sweep(script, dataset, min_bs, default_lr,
                                  sweep_epochs, vram_safety,
                                  num_samples=num_samples, hidden_size=hidden_size)
    if not throughputs:
        print("Phase 1 failed — no successful batch size trials.")
        return {}
    best_bs = _select_batch_size(throughputs, thr_tolerance)
    print(f"  => Selected batch_size={best_bs} "
          f"({throughputs[best_bs]:.0f} samples/s, "
          f"peak={max(throughputs.values()):.0f})")

    # --- Phase 2: LR halving (two tracks) ---
    sgd_results = lr_halving(script, dataset, best_bs, sgd_lrs, "sgd", tune_epochs,
                             num_samples=num_samples, hidden_size=hidden_size)
    adam_results = lr_halving(script, dataset, best_bs, adam_lrs, "adam", tune_epochs,
                             num_samples=num_samples, hidden_size=hidden_size)

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
                           tune_epochs, selected["optimizer"], sched,
                           num_samples=num_samples, hidden_size=hidden_size)
        if trial:
            print(f"acc={trial['accuracy']:.2f}%  {trial['throughput']:.0f} samples/s")
            scheduler_cmp[sched] = trial
        else:
            print("FAILED")

    # Pick best scheduler by accuracy (throughput impact is negligible)
    best_sched = "none"
    if scheduler_cmp:
        best_sched = max(scheduler_cmp, key=lambda s: scheduler_cmp[s]["accuracy"])

    best_params = {
        "batch_size": best_bs,
        "learning_rate": selected["lr"],
        "optimizer": selected["optimizer"],
        "scheduler": best_sched,
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
