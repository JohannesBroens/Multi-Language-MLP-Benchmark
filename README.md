# ML Benchmark: C vs Python MLP Implementations

A multi-language machine learning benchmark comparing MLP (Multi-Layer Perceptron) implementations in C, NumPy, and PyTorch. The same algorithm is implemented identically across languages to enable fair performance comparisons across CPU and GPU backends.

## MLP Architecture

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Hidden layers | 1 (64 neurons) | Simple enough to implement from scratch, sufficient for tabular data |
| Hidden activation | ReLU | Fast, avoids vanishing gradients |
| Output activation | Softmax | Produces class probabilities for multi-class classification |
| Loss | Cross-entropy | Standard for classification; clean gradient with softmax |
| Initialization | Xavier uniform (sqrt(2/fan_in)) | Keeps activation variance stable across layers |
| Optimizer | Mini-batch SGD (lr=0.01, batch=32) | Simple, no dependencies, easy to implement identically across languages |
| Epochs | 1000 | Sufficient convergence for all datasets |

## Implementations

| Implementation | File | Description |
|---------------|------|-------------|
| C (CPU) | `src/c/models/mlp/mlp_cpu.c` | Manual backprop in C99, compiled with OpenMP parallelization |
| C (CUDA) | `src/c/models/mlp/mlp.cu` | Same algorithm on GPU, one CUDA thread per sample, atomicAdd for gradients |
| NumPy (CPU) | `src/python/models/mlp/mlp_numpy.py` | Vectorized NumPy, exact replica of C algorithm |
| PyTorch (CPU) | `src/python/models/mlp/mlp_pytorch.py` | nn.Module with manual Xavier init to match C, CPU backend |
| PyTorch (CUDA) | `src/python/models/mlp/mlp_pytorch.py` | Same PyTorch model on GPU via `--device cuda` |

All 5 implementations produce identical standardized output for benchmark parsing.

## Datasets

| Name | Samples | Features | Classes | Source |
|------|---------|----------|---------|--------|
| `generated` | 1000 | 2 | 2 | Synthetic 2D circle classification |
| `iris` | 150 | 4 | 3 | UCI Iris |
| `wine-red` | 1599 | 11 | 11 | UCI Wine Quality (red) |
| `wine-white` | 4898 | 11 | 11 | UCI Wine Quality (white) |
| `breast-cancer` | 569 | 30 | 2 | Wisconsin Diagnostic Breast Cancer |

## Quick Start

### Prerequisites

- **C**: GCC (C99), CMake 3.10+, optionally CUDA Toolkit and OpenMP
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

### Build C (CPU with OpenMP)

```bash
cd src/c && mkdir -p build_cpu && cd build_cpu
cmake .. -DUSE_CUDA=OFF
make
```

### Build C (with CUDA)

```bash
cd src/c && mkdir -p build_cuda && cd build_cuda
cmake .. -DUSE_CUDA=ON
make
```

### Build Both (for benchmarking)

To run the full benchmark comparing all implementations, build both:

```bash
cd src/c
mkdir -p build_cpu && cd build_cpu && cmake .. -DUSE_CUDA=OFF && make && cd ..
mkdir -p build_cuda && cd build_cuda && cmake .. -DUSE_CUDA=ON && make && cd ..
```

### Run Individual Implementations

```bash
# C (CPU)
./src/c/build_cpu/main --dataset iris

# C (CUDA)
./src/c/build_cuda/main --dataset iris

# NumPy
python3 src/python/models/mlp/mlp_numpy.py --dataset iris

# PyTorch (CPU)
python3 src/python/models/mlp/mlp_pytorch.py --dataset iris --device cpu

# PyTorch (CUDA)
python3 src/python/models/mlp/mlp_pytorch.py --dataset iris --device cuda
```

### Run Full Benchmark

```bash
python3 src/scripts/benchmark.py --datasets generated,iris,breast-cancer --runs 3
```

This runs all available implementations, prints a markdown results table, and generates bar charts in `figs/`.

## Benchmark Results

Run `python3 src/scripts/benchmark.py` to regenerate. All 5 implementations on an RTX 3070:

| Dataset | C (CPU) | C (CUDA) | NumPy (CPU) | PyTorch (CPU) | PyTorch (CUDA) |
|---------|---------|----------|-------------|---------------|----------------|
| **generated** | 99.0% / 1.53s | 98.0% / 2.06s | 98.0% / 1.30s | 99.5% / 11.5s | 99.0% / 17.3s |
| **iris** | 100% / 0.36s | 100% / 0.41s | 100% / 0.23s | 96.7% / 2.18s | 93.3% / 3.51s |
| **breast-cancer** | 98.2% / 4.37s | 96.5% / 4.94s | 97.4% / 0.81s | 96.5% / 6.17s | 95.6% / 10.8s |

*Format: Accuracy / Train time. Results vary between runs due to random initialization and shuffling.*

Charts saved to `figs/`:

- `figs/benchmark_train_time.png` — Training time comparison
- `figs/benchmark_accuracy.png` — Test accuracy comparison

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
│   ├── python/
│   │   ├── models/mlp/
│   │   │   ├── data_utils.py      # Shared data loading (mirrors C data_loader)
│   │   │   ├── mlp_numpy.py       # NumPy implementation
│   │   │   └── mlp_pytorch.py     # PyTorch implementation
│   │   └── setup.py
│   └── scripts/
│       ├── benchmark.py           # Benchmark runner with chart generation
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
