# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-language MLP benchmark: identical algorithm implemented in C (CPU+OpenMP, CUDA), Rust (CPU+Rayon, cuBLAS, raw CUDA kernels), NumPy, and PyTorch. Compares performance across languages and backends on tabular datasets (iris, wine quality, breast cancer, synthetic).

## Build Commands

### C (CPU only, with OpenMP)
```bash
cd src/c && mkdir -p build_cpu && cd build_cpu
cmake .. -DUSE_CUDA=OFF
make
```

### C (with CUDA)
```bash
cd src/c && mkdir -p build_cuda && cd build_cuda
cmake .. -DUSE_CUDA=ON
make
```

### Rust (all targets)
```bash
cd src/rust && cargo build --release
```

### Rust (CPU only, no CUDA needed)
```bash
cd src/rust && cargo build --release -p mlp-cpu
```

### Run Implementations
```bash
# C CPU (from project root)
./src/c/build_cpu/main --dataset <name>

# C CUDA
./src/c/build_cuda/main --dataset <name>

# Rust CPU
./src/rust/target/release/mlp-cpu --dataset <name>

# Rust cuBLAS
./src/rust/target/release/mlp-cuda-cublas --dataset <name>

# Rust CUDA Kernels
./src/rust/target/release/mlp-cuda-kernels --dataset <name>

# NumPy
python3 src/python/models/mlp/mlp_numpy.py --dataset <name>

# PyTorch
python3 src/python/models/mlp/mlp_pytorch.py --dataset <name> --device cpu|cuda

# Datasets: generated, iris, wine-red, wine-white, breast-cancer
```

### Run Benchmark
```bash
python3 src/scripts/benchmark.py --datasets generated,iris,breast-cancer --runs 3
```

### Full Pipeline (download data, preprocess, build, run)
```bash
src/scripts/run_pipeline.sh [dataset_name]
```

### Download Datasets Only
```bash
bash src/scripts/download_datasets.sh
python3 src/scripts/preprocess_iris.py
```

## Architecture

### C Implementation

The C build uses CMake with a top-level `src/c/CMakeLists.txt` that adds model subdirectories. Each model (e.g., `models/mlp/`) has its own `CMakeLists.txt` producing a shared library linked into the `main` executable.

**Key flow:** `main.c` parses `--dataset` arg -> loads data -> z-score normalizes features -> 80/20 train/test split with shuffle -> trains MLP -> evaluates on test set -> prints timing benchmarks.

**CUDA/CPU split:** The `USE_CUDA` CMake option selects between `mlp.cu` (CUDA) and `mlp_cpu.c` (OpenMP) at build time. Both implement the same interface defined in `mlp.h`. The MLP struct holds weight matrices and bias vectors; the API is `mlp_initialize`, `mlp_train`, `mlp_evaluate`, `mlp_free`.

### Rust Implementation

Cargo workspace at `src/rust/` with a shared `mlp-common` crate (data loading, CLI, normalization) and three binary crates:

- **mlp-cpu**: Rayon threadpool (physical cores only) + cache-tiled GEMM (TILE=64, i-k-j order). Threshold guards skip parallelism for small matrices.
- **mlp-cuda-cublas**: cuBLAS for GEMM via raw FFI, custom CUDA kernels (.cu) for elementwise ops (bias+ReLU, softmax, CE loss, SGD). Row-major via transposition trick.
- **mlp-cuda-kernels**: All custom CUDA kernels including shared-memory tiled matmul (TILE_DIM=16). No cuBLAS dependency.

### Python Implementations

`src/python/models/mlp/data_utils.py` provides shared data loading that mirrors C's `data_loader.c`. Both `mlp_numpy.py` and `mlp_pytorch.py` import from it and produce the same standardized output format for benchmark parsing.

### MLP Details

Single hidden layer with ReLU activation, softmax output with cross-entropy loss. Xavier weight initialization: `Uniform(-sqrt(2/fan_in), sqrt(2/fan_in))`. Mini-batch SGD with gradient averaging (`lr_scaled = lr / batch_size`).

**Hyperparameters** are defined as macros in `mlp.h` (`NUM_EPOCHS=1000`, `LEARNING_RATE=0.01`). Hidden size (64) and batch size (32) are set in `main.c` and matched in Python scripts.

### Output Format

All implementations print identical standardized output for benchmark parsing:
```
=== Results on <dataset> ===
Test Loss:     0.1234
Test Accuracy: 96.67%
Train time:    0.123 s
Eval time:     0.001 s
```

## Standards

- C99 standard
- CMake 3.10+
- Rust 2021 edition, Cargo workspace
- Python 3.8+
- CUDA architecture auto-detected via `native` in `models/mlp/CMakeLists.txt` and `-arch=native` in Rust build.rs
