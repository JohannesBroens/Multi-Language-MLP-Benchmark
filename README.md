# ML Language Playground: Multi-Language MLP Benchmark

A multi-language machine learning benchmark comparing MLP (Multi-Layer Perceptron) implementations across C, Rust, and Python. The same algorithm is implemented identically in 8 variants spanning CPU and GPU backends to measure throughput scaling across dataset sizes, batch sizes, and hidden layer widths.

## MLP Architecture

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Hidden layers | 1 (configurable size, default 64) | Simple enough to implement from scratch, sufficient for tabular data |
| Hidden activation | ReLU | Fast, avoids vanishing gradients |
| Output activation | Softmax | Produces class probabilities for multi-class classification |
| Loss | Cross-entropy | Standard for classification; clean gradient with softmax |
| Initialization | Xavier uniform (sqrt(2/fan_in)) | Keeps activation variance stable across layers |
| Optimizer | Mini-batch SGD (configurable lr/batch) | Simple, no dependencies, easy to implement identically across languages |

## Implementations

| Implementation | File | Description |
|---------------|------|-------------|
| C (CPU) | `src/c/models/mlp/mlp_cpu.c` | Manual backprop in C99 with OpenMP parallelization and cache-tiled GEMM |
| C (CUDA) | `src/c/models/mlp/mlp.cu` | GPU kernels, one CUDA thread per sample, atomicAdd for gradients |
| Rust (CPU) | `src/rust/mlp-cpu/src/main.rs` | Rayon threadpool (physical cores) + cache-tiled GEMM (TILE=64) |
| Rust (cuBLAS) | `src/rust/mlp-cuda-cublas/src/main.rs` | cuBLAS FFI for GEMM + custom CUDA kernels for elementwise ops |
| Rust (CUDA Kernels) | `src/rust/mlp-cuda-kernels/src/main.rs` | All custom CUDA kernels including shared-memory tiled matmul |
| NumPy (CPU) | `src/python/models/mlp/mlp_numpy.py` | Vectorized NumPy, exact replica of C algorithm |
| PyTorch (CPU) | `src/python/models/mlp/mlp_pytorch.py` | nn.Module with manual Xavier init to match C, CPU backend |
| PyTorch (CUDA) | `src/python/models/mlp/mlp_pytorch.py` | Same PyTorch model on GPU via `--device cuda` |

All 8 implementations produce identical standardized output for benchmark parsing, including throughput in samples/s.

## Datasets

| Name | Samples | Features | Classes | Source |
|------|---------|----------|---------|--------|
| `generated` | Configurable (default 1000) | 2 | 2 | Synthetic 2D circle classification |
| `iris` | 150 | 4 | 3 | UCI Iris |
| `wine-red` | 1599 | 11 | 11 | UCI Wine Quality (red) |
| `wine-white` | 4898 | 11 | 11 | UCI Wine Quality (white) |
| `breast-cancer` | 569 | 30 | 2 | Wisconsin Diagnostic Breast Cancer |

## Quick Start

### Prerequisites

- **C**: GCC (C99), CMake 3.10+, optionally CUDA Toolkit and OpenMP
- **Rust**: Cargo (2021 edition), optionally CUDA Toolkit for GPU crates
- **Python**: Python 3.8+, NumPy, matplotlib
- **Optional**: PyTorch (for `mlp_pytorch.py`)

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Download and Preprocess Datasets

```bash
bash src/scripts/download_datasets.sh
python3 src/scripts/preprocess_iris.py
```

### Build C

```bash
# CPU with OpenMP
cd src/c && mkdir -p build_cpu && cd build_cpu
cmake .. -DUSE_CUDA=OFF && make

# CUDA
cd src/c && mkdir -p build_cuda && cd build_cuda
cmake .. -DUSE_CUDA=ON && make
```

### Build Rust

```bash
# All targets (CPU + cuBLAS + CUDA Kernels)
cd src/rust && cargo build --release

# CPU only (no CUDA needed)
cd src/rust && cargo build --release -p mlp-cpu
```

### Run Individual Implementations

All implementations accept `--batch-size`, `--num-samples`, `--hidden-size`, and `--epochs` flags for configurable hyperparameters.

```bash
# C (CPU)
./src/c/build_cpu/main --dataset iris

# C (CUDA)
./src/c/build_cuda/main --dataset iris

# Rust (CPU)
./src/rust/target/release/mlp-cpu --dataset iris

# Rust (cuBLAS)
./src/rust/target/release/mlp-cuda-cublas --dataset iris

# Rust (CUDA Kernels)
./src/rust/target/release/mlp-cuda-kernels --dataset iris

# NumPy
python3 src/python/models/mlp/mlp_numpy.py --dataset iris

# PyTorch (CPU)
python3 src/python/models/mlp/mlp_pytorch.py --dataset iris --device cpu

# PyTorch (CUDA)
python3 src/python/models/mlp/mlp_pytorch.py --dataset iris --device cuda
```

### Run Benchmarks

```bash
# Standard mode: accuracy + train time on real datasets
python3 src/scripts/benchmark.py --mode standard --datasets generated,iris,breast-cancer --runs 3

# Scaling mode: throughput vs dataset size, batch size, and hidden size
python3 src/scripts/benchmark.py --mode scaling --runs 1
```

## Scaling Benchmark Results

Run `python3 src/scripts/benchmark.py --mode scaling` to regenerate. All 8 implementations on an NVIDIA RTX 3070:

### Peak Throughput (samples/s)

| Experiment | C (CUDA) | Rust (cuBLAS) | Rust (Kernels) | C (CPU) | NumPy | PyTorch (CUDA) | Rust (CPU) | PyTorch (CPU) |
|---|---|---|---|---|---|---|---|---|
| Dataset Size | **10.23M** | 10.21M | 8.40M | 5.64M | 4.11M | 3.66M | 3.47M | 2.40M |
| Batch Size | **17.40M** | 14.32M | 8.77M | 6.92M | 4.38M | 9.31M | 3.66M | 3.54M |
| Hidden Size | 9.71M | **10.26M** | 8.41M | 7.09M | 4.86M | 3.06M | 4.17M | 2.48M |

Rust cuBLAS matches or beats C CUDA across all experiments, winning on hidden size scaling.

### Scaling Plots

![Scaling Overview](figs/scaling_overview.png)

| Plot | Description |
|------|-------------|
| `figs/scaling_dataset_size.png` | Throughput vs dataset size (4K-1M samples) |
| `figs/scaling_batch_size.png` | Throughput vs batch size (128-16K) |
| `figs/scaling_hidden_size.png` | Throughput vs hidden layer width (32-1024) |
| `figs/gpu_speedup_dataset_size.png` | GPU/CPU speedup ratios vs dataset size |
| `figs/gpu_speedup_batch_size.png` | GPU/CPU speedup ratios vs batch size |
| `figs/gpu_speedup_hidden_size.png` | GPU/CPU speedup ratios vs hidden size |
| `figs/scaling_overview.png` | Combined overview of all scaling experiments |

## Project Structure

```
ML-in-C/
├── data/                          # Datasets (downloaded via scripts)
├── figs/                          # Generated benchmark charts
├── src/
│   ├── c/
│   │   ├── CMakeLists.txt         # Top-level CMake config
│   │   ├── data_loader.c/.h       # Dataset loaders (C)
│   │   ├── main.c                 # CLI entry point: load, normalize, train, evaluate
│   │   └── models/mlp/
│   │       ├── CMakeLists.txt     # MLP library build config
│   │       ├── mlp.h              # Shared interface and hyperparameters
│   │       ├── mlp_cpu.c          # CPU implementation (OpenMP)
│   │       └── mlp.cu             # CUDA implementation
│   ├── rust/
│   │   ├── Cargo.toml             # Workspace root
│   │   ├── mlp-common/            # Shared data loading, CLI, normalization
│   │   ├── mlp-cpu/               # CPU: Rayon + tiled GEMM
│   │   ├── mlp-cuda-cublas/       # GPU: cuBLAS FFI + custom CUDA kernels
│   │   └── mlp-cuda-kernels/      # GPU: all custom CUDA kernels (no cuBLAS)
│   ├── python/
│   │   ├── models/mlp/
│   │   │   ├── data_utils.py      # Shared data loading (mirrors C data_loader)
│   │   │   ├── mlp_numpy.py       # NumPy implementation
│   │   │   └── mlp_pytorch.py     # PyTorch implementation (CPU + CUDA)
│   │   └── setup.py
│   └── scripts/
│       ├── benchmark.py           # Benchmark runner (standard + scaling modes)
│       ├── tune_benchmark.py      # OMP threshold tuning script
│       ├── download_datasets.sh   # Download UCI datasets
│       ├── preprocess_iris.py     # Preprocess Iris data
│       └── run_pipeline.sh        # Full pipeline (download + build + run)
├── mathematical_foundations.md    # Math derivations for the MLP algorithm
├── requirements.txt               # Python dependencies
├── CLAUDE.md                      # AI assistant instructions
├── LICENSE                        # Apache 2.0
└── README.md
```

## Mathematical Foundations

See [mathematical_foundations.md](mathematical_foundations.md) for detailed derivations of:
- Feature normalization (z-score)
- Xavier weight initialization
- Forward propagation (ReLU + numerically stable softmax)
- Cross-entropy loss
- Backpropagation (including the softmax+CE gradient shortcut)
- Mini-batch SGD with gradient averaging

## License

This project is licensed under the Apache License (Version 2.0) — see the [LICENSE](LICENSE) file for details.
