"""Plotting module for ML-in-C benchmark results.

Extracted from benchmark.py. Handles all matplotlib chart generation for
scaling experiments (throughput, GPU speedup, overview) and standard mode
(bar charts for accuracy and train time).
"""
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
#  Plot styling constants
# ---------------------------------------------------------------------------

IMPL_STYLES = {
    "C (CUDA)":       {"color": "#2ecc71", "marker": "D", "linewidth": 3.0, "markersize": 10, "zorder": 10},
    "C (CPU)":        {"color": "#3498db", "marker": "o", "linewidth": 2.2, "markersize": 8,  "zorder": 5},
    "NumPy":          {"color": "#e67e22", "marker": "s", "linewidth": 2.0, "markersize": 7,  "zorder": 4},
    "PyTorch (CUDA)": {"color": "#9b59b6", "marker": "^", "linewidth": 2.2, "markersize": 8,  "zorder": 6},
    "PyTorch (CPU)":  {"color": "#e74c3c", "marker": "v", "linewidth": 1.8, "markersize": 7,  "zorder": 3},
    "Rust (CPU)":          {"color": "#f39c12", "marker": "P", "linewidth": 2.2, "markersize": 8,  "zorder": 5},
    "Rust (cuBLAS)":       {"color": "#1abc9c", "marker": "H", "linewidth": 2.8, "markersize": 9,  "zorder": 9},
    "Rust (CUDA Kernels)": {"color": "#e91e63", "marker": "X", "linewidth": 2.8, "markersize": 9,  "zorder": 8},
}

SPEEDUP_PAIRS = [
    ("C (CUDA)", "C (CPU)", "#2ecc71", "C CUDA / C CPU"),
    ("Rust (cuBLAS)", "Rust (CPU)", "#1abc9c", "Rust cuBLAS / Rust CPU"),
    ("Rust (CUDA Kernels)", "Rust (CPU)", "#e91e63", "Rust Kernels / Rust CPU"),
    ("PyTorch (CUDA)", "PyTorch (CPU)", "#9b59b6", "PyTorch CUDA / CPU"),
    ("Rust (cuBLAS)", "C (CUDA)", "#f39c12", "Rust cuBLAS / C CUDA"),
]

CNN_IMPL_STYLES = {
    "CNN C (CUDA)":            {"color": "#2ecc71", "marker": "D", "linewidth": 3.0, "markersize": 10, "zorder": 10},
    "CNN C (CPU)":             {"color": "#3498db", "marker": "o", "linewidth": 2.2, "markersize": 8,  "zorder": 5},
    "CNN PyTorch (CUDA)":      {"color": "#9b59b6", "marker": "^", "linewidth": 2.2, "markersize": 8,  "zorder": 6},
    "CNN PyTorch (CPU)":       {"color": "#e74c3c", "marker": "v", "linewidth": 1.8, "markersize": 7,  "zorder": 3},
    "CNN Rust (CPU)":          {"color": "#f39c12", "marker": "P", "linewidth": 2.2, "markersize": 8,  "zorder": 5},
    "CNN Rust (cuBLAS)":       {"color": "#1abc9c", "marker": "H", "linewidth": 2.8, "markersize": 9,  "zorder": 9},
    "CNN Rust (CUDA Kernels)": {"color": "#e91e63", "marker": "X", "linewidth": 2.8, "markersize": 9,  "zorder": 8},
}

CNN_SPEEDUP_PAIRS = [
    ("CNN C (CUDA)", "CNN C (CPU)", "#2ecc71", "C CUDA / C CPU"),
    ("CNN Rust (cuBLAS)", "CNN Rust (CPU)", "#1abc9c", "Rust cuBLAS / Rust CPU"),
    ("CNN Rust (CUDA Kernels)", "CNN Rust (CPU)", "#e91e63", "Rust Kernels / Rust CPU"),
    ("CNN PyTorch (CUDA)", "CNN PyTorch (CPU)", "#9b59b6", "PyTorch CUDA / CPU"),
    ("CNN Rust (cuBLAS)", "CNN C (CUDA)", "#f39c12", "Rust cuBLAS / C CUDA"),
]


# ---------------------------------------------------------------------------
#  Plot helpers
# ---------------------------------------------------------------------------

def _setup_style():
    """Dark-ish professional style."""
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "text.color": "#e0e0e0",
        "xtick.color": "#c0c0c0",
        "ytick.color": "#c0c0c0",
        "grid.color": "#334466",
        "grid.alpha": 0.5,
        "legend.facecolor": "#1a1a2e",
        "legend.edgecolor": "#555555",
        "legend.fontsize": 10,
        "font.family": "sans-serif",
        "figure.dpi": 150,
    })


def _ranked_legend(ax, throughput_data, labels):
    """Create legend ordered by peak throughput (best first)."""
    handles, leg_labels = ax.get_legend_handles_labels()
    peak = {}
    for label in labels:
        vals = [v for v in throughput_data.get(label, []) if v is not None]
        peak[label] = max(vals) if vals else 0

    paired = sorted(zip(handles, leg_labels), key=lambda x: peak.get(x[1], 0), reverse=True)
    if paired:
        handles, leg_labels = zip(*paired)
        ax.legend(handles, leg_labels, loc="best")


def _format_y_throughput(ax):
    """Format y-axis as readable throughput (K/s, M/s)."""
    def fmt(x, _):
        if x >= 1e6:
            return f"{x/1e6:.0f}M"
        if x >= 1e3:
            return f"{x/1e3:.0f}K"
        return f"{x:.0f}"
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))


def _format_x_samples(ax):
    """Format x-axis sample counts as readable (K, M)."""
    def fmt(x, _):
        if x >= 1e6:
            return f"{x/1e6:.0f}M"
        if x >= 1e3:
            return f"{x/1e3:.0f}K"
        return f"{x:.0f}"
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))


def _fit_gp(cache, cache_key, label, x_values=None):
    """Fit a Gaussian Process in log-log space for a single implementation.

    Returns (x_dense, y_mean, y_lower, y_upper, x_obs, y_obs_mean) or None
    if fewer than 3 data points are available.
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    except ImportError:
        return None

    if cache is None or cache_key is None:
        return None

    exp_data = cache.get(cache_key, {}).get(label, {})
    if not exp_data:
        return None

    x_obs_list = []
    y_obs_means = []
    x_all = []
    y_all = []
    for val_str, entry in sorted(exp_data.items(), key=lambda kv: float(kv[0])):
        samples = entry.get("samples", [])
        if not samples:
            continue
        x_val = float(val_str)
        mean_val = float(np.mean(samples))
        if mean_val <= 0 or x_val <= 0:
            continue
        x_obs_list.append(x_val)
        y_obs_means.append(mean_val)
        for s in samples:
            if s > 0:
                x_all.append(np.log2(x_val))
                y_all.append(np.log10(s))

    if len(x_obs_list) < 3:
        return None

    X_train = np.array(x_all).reshape(-1, 1)
    y_train = np.array(y_all)

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)
    try:
        gp.fit(X_train, y_train)
    except Exception:
        return None

    x_log_min = np.log2(min(x_obs_list))
    x_log_max = np.log2(max(x_obs_list))
    X_dense_log = np.linspace(x_log_min, x_log_max, 200).reshape(-1, 1)
    y_pred, y_std = gp.predict(X_dense_log, return_std=True)

    x_dense = 2 ** X_dense_log.ravel()
    y_mean = 10 ** y_pred
    y_lower = 10 ** (y_pred - 1.96 * y_std)
    y_upper = 10 ** (y_pred + 1.96 * y_std)

    return (x_dense, y_mean, y_lower, y_upper,
            np.array(x_obs_list), np.array(y_obs_means))


def _plot_throughput(ax, xs, throughput_data, available_labels, title, xlabel,
                     cache=None, cache_key=None, styles=None):
    """Plot throughput lines with implementation-specific styling.

    When cache/cache_key are provided, attempts GP regression for smooth curves
    with confidence bands. Falls back to direct line plot if GP fails.
    """
    active_styles = styles or IMPL_STYLES
    for label in available_labels:
        vals = throughput_data.get(label, [])
        xf = [xs[i] for i, v in enumerate(vals) if v is not None]
        yf = [v for v in vals if v is not None]
        if not yf:
            continue

        s = active_styles.get(label, {"color": "gray", "marker": ".", "linewidth": 1.5, "markersize": 6, "zorder": 1})

        gp_result = _fit_gp(cache, cache_key, label) if cache is not None else None

        if gp_result is not None:
            x_dense, y_mean, y_lower, y_upper, x_obs, y_obs_mean = gp_result
            ax.plot(x_dense, y_mean, color=s["color"],
                    linewidth=s["linewidth"], zorder=s["zorder"], label=label)
            ax.fill_between(x_dense, y_lower, y_upper,
                            color=s["color"], alpha=0.18, zorder=s["zorder"] - 1)
            ax.scatter(x_obs, y_obs_mean, marker=s["marker"], color=s["color"],
                       s=s["markersize"] ** 2, edgecolors="white", linewidths=0.8,
                       zorder=s["zorder"] + 1)
        else:
            ax.plot(xf, yf, marker=s["marker"], color=s["color"],
                    linewidth=s["linewidth"], markersize=s["markersize"],
                    zorder=s["zorder"], label=label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Throughput (samples/s)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    _format_y_throughput(ax)
    _ranked_legend(ax, throughput_data, available_labels)


def _speedup_with_ci(cache, cache_key, gpu_label, cpu_label):
    """Compute speedup with confidence interval using separate GP fits.

    Returns (x_points, speedup_mean, speedup_lower, speedup_upper) or None.
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    except ImportError:
        return None

    if cache is None or cache_key is None:
        return None

    gpu_data = cache.get(cache_key, {}).get(gpu_label, {})
    cpu_data = cache.get(cache_key, {}).get(cpu_label, {})
    if not gpu_data or not cpu_data:
        return None

    def _build_training_data(impl_data):
        x_all, y_all = [], []
        for val_str, entry in impl_data.items():
            samples = entry.get("samples", [])
            x_val = float(val_str)
            if x_val <= 0:
                continue
            for s in samples:
                if s > 0:
                    x_all.append(np.log2(x_val))
                    y_all.append(np.log10(s))
        return np.array(x_all), np.array(y_all)

    X_gpu, y_gpu = _build_training_data(gpu_data)
    X_cpu, y_cpu = _build_training_data(cpu_data)
    if len(set(X_gpu)) < 3 or len(set(X_cpu)) < 3:
        return None

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp_gpu = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)
    gp_cpu = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-6)
    try:
        gp_gpu.fit(X_gpu.reshape(-1, 1), y_gpu)
        gp_cpu.fit(X_cpu.reshape(-1, 1), y_cpu)
    except Exception:
        return None

    x_min = max(X_gpu.min(), X_cpu.min())
    x_max = min(X_gpu.max(), X_cpu.max())
    if x_min >= x_max:
        return None

    X_dense = np.linspace(x_min, x_max, 200).reshape(-1, 1)
    gpu_mean, gpu_std = gp_gpu.predict(X_dense, return_std=True)
    cpu_mean, cpu_std = gp_cpu.predict(X_dense, return_std=True)

    diff_mean = gpu_mean - cpu_mean
    diff_std = np.sqrt(gpu_std ** 2 + cpu_std ** 2)

    x_points = 2 ** X_dense.ravel()
    speedup_mean = 10 ** diff_mean
    speedup_lower = 10 ** (diff_mean - 1.96 * diff_std)
    speedup_upper = 10 ** (diff_mean + 1.96 * diff_std)

    return x_points, speedup_mean, speedup_lower, speedup_upper


# ---------------------------------------------------------------------------
#  Cache-to-throughput reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_throughput(cache, cache_key, x_values, labels):
    """Rebuild {label: [mean_or_None]} from cache for plotting."""
    throughput = {label: [] for label in labels}
    exp_data = cache.get(cache_key, {}) if cache else {}

    for val in x_values:
        val_str = str(val)
        for label in labels:
            entry = exp_data.get(label, {}).get(val_str, {})
            samples = entry.get("samples", []) if isinstance(entry, dict) else []
            if samples:
                throughput[label].append(float(np.mean(samples)))
            else:
                throughput[label].append(None)

    return throughput


def _discover_x_values(cache, cache_key):
    """Discover all unique x-values from cache for a given experiment."""
    exp_data = cache.get(cache_key, {}) if cache else {}
    x_set = set()
    for impl_data in exp_data.values():
        if isinstance(impl_data, dict):
            for val_str in impl_data.keys():
                try:
                    x_set.add(int(val_str))
                except ValueError:
                    pass
    return sorted(x_set)


def _cache_key(model, experiment):
    """Build a cache key like 'mlp/dataset_size'."""
    return f"{model}/{experiment}"


# ---------------------------------------------------------------------------
#  High-level plotting functions
# ---------------------------------------------------------------------------

def _get_output_dir(model, config=None):
    """Determine output directory for plots."""
    if config:
        base = config.get("plotting", {}).get("output_dir", "results/plots")
    else:
        base = "results/plots"
    return os.path.join(PROJECT_ROOT, base, model)


def plot_scaling_results(cache, cfg, model, labels, figs_dir=None):
    """Generate all scaling throughput plots from cached data.

    Args:
        cache: loaded benchmark cache dict
        cfg: scaling config dict with epochs, fixed_batch_size, etc.
        model: "mlp" or "cnn"
        labels: list of implementation labels
        figs_dir: output directory (default: results/plots/{model}/)
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available -- scaling mode requires matplotlib")
        return

    _setup_style()
    figs_dir = figs_dir or _get_output_dir(model)
    os.makedirs(figs_dir, exist_ok=True)

    styles = CNN_IMPL_STYLES if model == "cnn" else IMPL_STYLES
    prefix = f"{model.upper()}: " if model != "mlp" else ""
    epochs = cfg["epochs"]

    if model == "cnn":
        batch_sizes = _discover_x_values(cache, _cache_key(model, "batch_size"))
        if batch_sizes:
            tp_bs = _reconstruct_throughput(
                cache, _cache_key(model, "batch_size"), batch_sizes, labels)
            fig, ax = plt.subplots(figsize=(12, 7))
            _plot_throughput(ax, batch_sizes, tp_bs, labels,
                             f"{prefix}Throughput vs Batch Size\nMNIST (60K samples), epochs={epochs}",
                             "Batch Size",
                             cache=cache, cache_key=_cache_key(model, "batch_size"),
                             styles=styles)
            plt.tight_layout()
            p = os.path.join(figs_dir, "scaling_batch_size.png")
            plt.savefig(p); plt.close()
            print(f"  Saved: {p}")
    else:
        fixed_bs = cfg.get("fixed_batch_size", 4096)
        fixed_ns = cfg.get("fixed_num_samples", 262144)

        # Dataset size
        dataset_sizes = _discover_x_values(cache, _cache_key(model, "dataset_size"))
        if dataset_sizes:
            tp_ds = _reconstruct_throughput(
                cache, _cache_key(model, "dataset_size"), dataset_sizes, labels)
            fig, ax = plt.subplots(figsize=(12, 7))
            _plot_throughput(ax, dataset_sizes, tp_ds, labels,
                             f"{prefix}Throughput vs Dataset Size\nbatch_size={fixed_bs}, epochs={epochs}",
                             "Dataset Size (samples)",
                             cache=cache, cache_key=_cache_key(model, "dataset_size"),
                             styles=styles)
            _format_x_samples(ax)
            plt.tight_layout()
            p = os.path.join(figs_dir, "scaling_dataset_size.png")
            plt.savefig(p); plt.close()
            print(f"  Saved: {p}")

        # Batch size
        batch_sizes = _discover_x_values(cache, _cache_key(model, "batch_size"))
        if batch_sizes:
            tp_bs = _reconstruct_throughput(
                cache, _cache_key(model, "batch_size"), batch_sizes, labels)
            fig, ax = plt.subplots(figsize=(12, 7))
            _plot_throughput(ax, batch_sizes, tp_bs, labels,
                             f"{prefix}Throughput vs Batch Size\nnum_samples={fixed_ns:,}, epochs={epochs}",
                             "Batch Size",
                             cache=cache, cache_key=_cache_key(model, "batch_size"),
                             styles=styles)
            plt.tight_layout()
            p = os.path.join(figs_dir, "scaling_batch_size.png")
            plt.savefig(p); plt.close()
            print(f"  Saved: {p}")

        # Hidden size
        hidden_sizes = _discover_x_values(cache, _cache_key(model, "hidden_size"))
        if hidden_sizes:
            tp_hs = _reconstruct_throughput(
                cache, _cache_key(model, "hidden_size"), hidden_sizes, labels)
            fig, ax = plt.subplots(figsize=(12, 7))
            _plot_throughput(ax, hidden_sizes, tp_hs, labels,
                             f"{prefix}Throughput vs Hidden Layer Size\nsamples={fixed_ns:,}, batch={fixed_bs}, epochs={epochs}",
                             "Hidden Size",
                             cache=cache, cache_key=_cache_key(model, "hidden_size"),
                             styles=styles)
            plt.tight_layout()
            p = os.path.join(figs_dir, "scaling_hidden_size.png")
            plt.savefig(p); plt.close()
            print(f"  Saved: {p}")


def plot_speedup_results(cache, cfg, model, labels, figs_dir=None):
    """Generate GPU speedup plots from cached data."""
    if not HAS_MATPLOTLIB:
        return

    _setup_style()
    figs_dir = figs_dir or _get_output_dir(model)
    os.makedirs(figs_dir, exist_ok=True)

    styles = CNN_IMPL_STYLES if model == "cnn" else IMPL_STYLES
    speedup_pairs = CNN_SPEEDUP_PAIRS if model == "cnn" else SPEEDUP_PAIRS
    prefix = f"{model.upper()}: " if model != "mlp" else ""

    # Build experiments list from cache
    experiments = []
    if model != "cnn":
        dataset_sizes = _discover_x_values(cache, _cache_key(model, "dataset_size"))
        if dataset_sizes:
            tp_ds = _reconstruct_throughput(
                cache, _cache_key(model, "dataset_size"), dataset_sizes, labels)
            experiments.append(("dataset_size", dataset_sizes, tp_ds, "Dataset Size (samples)", True))

    batch_sizes = _discover_x_values(cache, _cache_key(model, "batch_size"))
    if batch_sizes:
        tp_bs = _reconstruct_throughput(
            cache, _cache_key(model, "batch_size"), batch_sizes, labels)
        experiments.append(("batch_size", batch_sizes, tp_bs, "Batch Size", False))

    if model != "cnn":
        hidden_sizes = _discover_x_values(cache, _cache_key(model, "hidden_size"))
        if hidden_sizes:
            tp_hs = _reconstruct_throughput(
                cache, _cache_key(model, "hidden_size"), hidden_sizes, labels)
            experiments.append(("hidden_size", hidden_sizes, tp_hs, "Hidden Size", False))

    for exp_name, xs, tp_data, xlabel, use_sample_fmt in experiments:
        fig, ax = plt.subplots(figsize=(12, 7))
        sp_cache_key = _cache_key(model, exp_name)

        any_plotted = False
        for gpu_label, cpu_label, color, legend_name in speedup_pairs:
            gpu_vals = tp_data.get(gpu_label, [])
            cpu_vals = tp_data.get(cpu_label, [])
            if not gpu_vals or not cpu_vals:
                continue

            xf, yf = [], []
            for i in range(min(len(gpu_vals), len(cpu_vals))):
                if gpu_vals[i] is not None and cpu_vals[i] is not None and cpu_vals[i] > 0:
                    xf.append(xs[i])
                    yf.append(gpu_vals[i] / cpu_vals[i])

            sp_result = _speedup_with_ci(cache, sp_cache_key, gpu_label, cpu_label) if cache else None

            if sp_result is not None:
                x_pts, sp_mean, sp_lower, sp_upper = sp_result
                ax.plot(x_pts, sp_mean, color=color, linewidth=2.5, label=legend_name)
                ax.fill_between(x_pts, sp_lower, sp_upper, color=color, alpha=0.20)
                if yf:
                    ax.scatter(xf, yf, color=color, marker="o", s=64,
                               edgecolors="white", linewidths=0.8, zorder=10)
                any_plotted = True
            elif yf:
                ax.plot(xf, yf, marker="o", color=color, linewidth=2.5,
                        markersize=8, label=legend_name)
                any_plotted = True

        if not any_plotted:
            plt.close()
            continue

        ax.axhline(y=1.0, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1.5, label="Break-even (1x)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Speedup (x)")
        ax.set_title(f"{prefix}GPU Speedup vs {xlabel}\n(higher = GPU wins)", fontsize=14, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)

        if use_sample_fmt:
            _format_x_samples(ax)

        handles, leg_labels = ax.get_legend_handles_labels()
        peak_sp = {}
        for h, ll in zip(handles, leg_labels):
            if ll == "Break-even (1x)":
                peak_sp[ll] = -1
            else:
                ydata = h.get_ydata()
                peak_sp[ll] = max(ydata) if len(ydata) > 0 else 0
        paired = sorted(zip(handles, leg_labels), key=lambda x: peak_sp.get(x[1], 0), reverse=True)
        handles, leg_labels = zip(*paired)
        ax.legend(handles, leg_labels, loc="best")

        plt.tight_layout()
        p = os.path.join(figs_dir, f"gpu_speedup_{exp_name}.png")
        plt.savefig(p); plt.close()
        print(f"  Saved: {p}")


def plot_overview(cache, cfg, model, labels, figs_dir=None):
    """Generate combined overview plot."""
    if not HAS_MATPLOTLIB:
        return

    _setup_style()
    figs_dir = figs_dir or _get_output_dir(model)
    os.makedirs(figs_dir, exist_ok=True)

    styles = CNN_IMPL_STYLES if model == "cnn" else IMPL_STYLES
    speedup_pairs = CNN_SPEEDUP_PAIRS if model == "cnn" else SPEEDUP_PAIRS
    epochs = cfg["epochs"]

    if model == "cnn":
        batch_sizes = _discover_x_values(cache, _cache_key(model, "batch_size"))
        if not batch_sizes:
            return
        tp_bs = _reconstruct_throughput(
            cache, _cache_key(model, "batch_size"), batch_sizes, labels)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        _plot_throughput(axes[0], batch_sizes, tp_bs, labels,
                         "Batch Size Scaling (MNIST)", "Batch Size",
                         cache=cache, cache_key=_cache_key(model, "batch_size"),
                         styles=styles)

        bs_cache_key = _cache_key(model, "batch_size")
        for gpu_label, cpu_label, color, legend_name in speedup_pairs[:3]:
            sp_result = _speedup_with_ci(cache, bs_cache_key, gpu_label, cpu_label) if cache else None
            if sp_result is not None:
                x_pts, sp_mean, sp_lower, sp_upper = sp_result
                axes[1].plot(x_pts, sp_mean, color=color, linewidth=2.5, label=legend_name)
                axes[1].fill_between(x_pts, sp_lower, sp_upper, color=color, alpha=0.15)

        axes[1].axhline(y=1.0, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1.5)
        axes[1].set_xscale("log", base=2)
        axes[1].set_xlabel("Batch Size")
        axes[1].set_ylabel("Speedup (x)")
        axes[1].set_title("GPU Speedup vs Batch Size", fontweight="bold")
        axes[1].grid(True, which="both", alpha=0.3)
        handles, leg_labels = axes[1].get_legend_handles_labels()
        if handles:
            axes[1].legend(loc="best")

        fig.suptitle("CNN Benchmark: GPU Scaling Analysis (LeNet-5 / MNIST)",
                      fontsize=18, fontweight="bold", y=0.98)
    else:
        fixed_bs = cfg.get("fixed_batch_size", 4096)
        fixed_ns = cfg.get("fixed_num_samples", 262144)

        dataset_sizes = _discover_x_values(cache, _cache_key(model, "dataset_size"))
        batch_sizes = _discover_x_values(cache, _cache_key(model, "batch_size"))
        hidden_sizes = _discover_x_values(cache, _cache_key(model, "hidden_size"))

        tp_ds = _reconstruct_throughput(
            cache, _cache_key(model, "dataset_size"), dataset_sizes, labels) if dataset_sizes else None
        tp_bs = _reconstruct_throughput(
            cache, _cache_key(model, "batch_size"), batch_sizes, labels) if batch_sizes else None
        tp_hs = _reconstruct_throughput(
            cache, _cache_key(model, "hidden_size"), hidden_sizes, labels) if hidden_sizes else None

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        if dataset_sizes and tp_ds:
            _plot_throughput(axes[0, 0], dataset_sizes, tp_ds, labels,
                             f"Dataset Size Scaling (batch={fixed_bs})", "Dataset Size",
                             cache=cache, cache_key=_cache_key(model, "dataset_size"),
                             styles=styles)
            _format_x_samples(axes[0, 0])
        else:
            axes[0, 0].set_visible(False)

        if batch_sizes and tp_bs:
            _plot_throughput(axes[0, 1], batch_sizes, tp_bs, labels,
                             f"Batch Size Scaling (samples={fixed_ns:,})", "Batch Size",
                             cache=cache, cache_key=_cache_key(model, "batch_size"),
                             styles=styles)
        else:
            axes[0, 1].set_visible(False)

        if hidden_sizes and tp_hs:
            _plot_throughput(axes[1, 0], hidden_sizes, tp_hs, labels,
                             f"Hidden Size Scaling (batch={fixed_bs})", "Hidden Size",
                             cache=cache, cache_key=_cache_key(model, "hidden_size"),
                             styles=styles)

            hs_cache_key = _cache_key(model, "hidden_size")
            for gpu_label, cpu_label, color, legend_name in speedup_pairs[:3]:
                sp_result = _speedup_with_ci(cache, hs_cache_key, gpu_label, cpu_label) if cache else None
                if sp_result is not None:
                    x_pts, sp_mean, sp_lower, sp_upper = sp_result
                    axes[1, 1].plot(x_pts, sp_mean, color=color, linewidth=2.5, label=legend_name)
                    axes[1, 1].fill_between(x_pts, sp_lower, sp_upper, color=color, alpha=0.15)

            axes[1, 1].axhline(y=1.0, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1.5)
            axes[1, 1].set_xscale("log", base=2)
            axes[1, 1].set_xlabel("Hidden Size")
            axes[1, 1].set_ylabel("Speedup (x)")
            axes[1, 1].set_title("GPU Speedup vs Hidden Size", fontweight="bold")
            axes[1, 1].grid(True, which="both", alpha=0.3)
            handles, leg_labels = axes[1, 1].get_legend_handles_labels()
            if handles:
                axes[1, 1].legend(loc="best")
        else:
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)

        fig.suptitle("MLP Benchmark: GPU Scaling Analysis", fontsize=18, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(figs_dir, "scaling_overview.png")
    plt.savefig(p); plt.close()
    print(f"  Saved: {p}")


def generate_charts(results, figs_dir, model="mlp"):
    """Generate standard mode bar charts (accuracy + train time)."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping chart generation")
        return

    _setup_style()
    os.makedirs(figs_dir, exist_ok=True)
    datasets = sorted(results.keys())
    all_labels = []
    for d in datasets:
        for label in results[d]:
            if label not in all_labels:
                all_labels.append(label)

    prefix = f"{model.upper()}: " if model != "mlp" else ""
    x = np.arange(len(datasets))
    width = 0.8 / max(len(all_labels), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(all_labels):
        vals = [results[d][label]["mean_train"] if label in results[d] else 0 for d in datasets]
        errs = [results[d][label]["std_train"] or 0 if label in results[d] else 0 for d in datasets]
        ax.bar(x + i * width, vals, width, yerr=errs, label=label, capsize=3)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Train Time (s)")
    ax.set_title(f"{prefix}Training Time by Implementation")
    ax.set_xticks(x + width * (len(all_labels) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "benchmark_train_time.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(all_labels):
        vals = [results[d][label]["mean_acc"] if label in results[d] else 0 for d in datasets]
        errs = [results[d][label]["std_acc"] or 0 if label in results[d] else 0 for d in datasets]
        ax.bar(x + i * width, vals, width, yerr=errs, label=label, capsize=3)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"{prefix}Test Accuracy by Implementation")
    ax.set_xticks(x + width * (len(all_labels) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "benchmark_accuracy.png"), dpi=150)
    plt.close()
    print(f"\nCharts saved to {figs_dir}/")


def print_peak_summary(cache, model, labels):
    """Print peak throughput summary table to stdout."""
    print("\n" + "=" * 70)
    print("  PEAK THROUGHPUT SUMMARY")
    print("=" * 70)

    experiments = []
    if model != "cnn":
        ds = _discover_x_values(cache, _cache_key(model, "dataset_size"))
        if ds:
            tp = _reconstruct_throughput(cache, _cache_key(model, "dataset_size"), ds, labels)
            experiments.append(("Dataset Size", tp))

    bs = _discover_x_values(cache, _cache_key(model, "batch_size"))
    if bs:
        tp = _reconstruct_throughput(cache, _cache_key(model, "batch_size"), bs, labels)
        experiments.append(("Batch Size", tp))

    if model != "cnn":
        hs = _discover_x_values(cache, _cache_key(model, "hidden_size"))
        if hs:
            tp = _reconstruct_throughput(cache, _cache_key(model, "hidden_size"), hs, labels)
            experiments.append(("Hidden Size", tp))

    for exp_name, tp_data in experiments:
        print(f"\n  {exp_name} experiment:")
        ranked = []
        for label in labels:
            vals = [v for v in tp_data.get(label, []) if v is not None]
            if vals:
                ranked.append((label, max(vals)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        for i, (label, peak) in enumerate(ranked):
            bar = "#" * int(peak / ranked[0][1] * 40) if ranked else ""
            if peak >= 1e6:
                print(f"    {i+1}. {label:<25} {peak/1e6:>8.2f}M/s  {bar}")
            else:
                print(f"    {i+1}. {label:<25} {peak/1e3:>8.0f}K/s  {bar}")


def regenerate_all_plots(config, model):
    """Regenerate all plots from cached data — called by pipeline.py."""
    import json

    cache_file = os.path.join(PROJECT_ROOT, "results", "cache", "benchmark_cache.json")
    if not os.path.isfile(cache_file):
        print(f"No cache file found at {cache_file}")
        return

    with open(cache_file) as f:
        cache = json.load(f)

    # Discover labels from cache
    labels = set()
    for exp_key, exp_data in cache.items():
        if exp_key.startswith(f"{model}/"):
            for label in exp_data:
                labels.add(label)
    labels = sorted(labels)

    if not labels:
        print(f"No cached data found for model '{model}'")
        return

    cfg = config.get("scaling", {})
    figs_dir = _get_output_dir(model, config)

    print(f"Regenerating plots for {model.upper()} -> {figs_dir}")
    plot_scaling_results(cache, cfg, model, labels, figs_dir)
    plot_speedup_results(cache, cfg, model, labels, figs_dir)
    plot_overview(cache, cfg, model, labels, figs_dir)
    print_peak_summary(cache, model, labels)
