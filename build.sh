#!/bin/bash
set -e

# ─────────────────────────────────────────────────────────────
#  Build script for ML Language Playground
#  Detects available toolchains and builds all possible targets
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

built=()
skipped=()

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[SKIP]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

# ─────────────────────────────────────────────────────────────
#  Detect toolchains
# ─────────────────────────────────────────────────────────────

HAS_CMAKE=false; command -v cmake  &>/dev/null && HAS_CMAKE=true
HAS_GCC=false;   command -v gcc    &>/dev/null && HAS_GCC=true
HAS_NVCC=false;  command -v nvcc   &>/dev/null && HAS_NVCC=true
HAS_CARGO=false; command -v cargo  &>/dev/null && HAS_CARGO=true
HAS_PYTHON=false

# Find Python (prefer venv)
PYTHON=""
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
    HAS_PYTHON=true
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
    HAS_PYTHON=true
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║      ML Language Playground — Build          ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
info "Detected toolchains:"
$HAS_CMAKE  && ok   "CMake   $(cmake --version | head -1 | grep -oP '[\d.]+')" || warn "CMake   — not found"
$HAS_GCC    && ok   "GCC     $(gcc -dumpversion)" || warn "GCC     — not found"
$HAS_NVCC   && ok   "CUDA    $(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+')" || warn "CUDA    — not found (GPU targets will be skipped)"
$HAS_CARGO  && ok   "Cargo   $(cargo --version | grep -oP '[\d.]+')" || warn "Cargo   — not found (Rust targets will be skipped)"
$HAS_PYTHON && ok   "Python  $($PYTHON --version 2>&1 | grep -oP '[\d.]+')" || warn "Python  — not found"
echo ""

# ─────────────────────────────────────────────────────────────
#  1. Download datasets
# ─────────────────────────────────────────────────────────────

info "Downloading datasets..."
if [ -f "$PROJECT_ROOT/src/scripts/download_datasets.sh" ]; then
    bash "$PROJECT_ROOT/src/scripts/download_datasets.sh"
    if $HAS_PYTHON; then
        $PYTHON "$PROJECT_ROOT/src/scripts/preprocess_iris.py"
    fi
    ok "Datasets ready"
else
    warn "Dataset scripts not found"
fi
echo ""

# ─────────────────────────────────────────────────────────────
#  2. Python dependencies
# ─────────────────────────────────────────────────────────────

if $HAS_PYTHON; then
    info "Installing Python dependencies..."
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        # Use uv if available, otherwise pip
        if command -v uv &>/dev/null && [ -d "$PROJECT_ROOT/.venv" ]; then
            uv pip install --python "$PROJECT_ROOT/.venv/bin/python" -r "$PROJECT_ROOT/requirements.txt" -q
        else
            $PYTHON -m pip install -r "$PROJECT_ROOT/requirements.txt" -q
        fi
        ok "Python dependencies installed"
        built+=("NumPy (CPU)" "PyTorch (CPU)")
        if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            built+=("PyTorch (CUDA)")
        else
            skipped+=("PyTorch (CUDA) — no GPU or torch.cuda unavailable")
        fi
    fi
else
    skipped+=("Python targets — python3 not found")
fi
echo ""

# ─────────────────────────────────────────────────────────────
#  3. C builds (CPU + CUDA)
# ─────────────────────────────────────────────────────────────

if $HAS_CMAKE && $HAS_GCC; then
    # CPU build
    info "Building C (CPU)..."
    mkdir -p "$PROJECT_ROOT/src/c/build_cpu"
    cd "$PROJECT_ROOT/src/c/build_cpu"
    cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
    make -j"$(nproc)" 2>&1 | tail -3
    ok "C (CPU) — src/c/build_cpu/main, src/c/build_cpu/cnn_main"
    built+=("C (CPU)")

    # CUDA build
    if $HAS_NVCC; then
        info "Building C (CUDA)..."
        mkdir -p "$PROJECT_ROOT/src/c/build_cuda"
        cd "$PROJECT_ROOT/src/c/build_cuda"
        cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
        make -j"$(nproc)" 2>&1 | tail -3
        ok "C (CUDA) — src/c/build_cuda/main, src/c/build_cuda/cnn_main"
        built+=("C (CUDA)")
    else
        skipped+=("C (CUDA) — nvcc not found")
    fi
else
    skipped+=("C targets — cmake or gcc not found")
fi
cd "$PROJECT_ROOT"
echo ""

# ─────────────────────────────────────────────────────────────
#  4. Rust builds
# ─────────────────────────────────────────────────────────────

if $HAS_CARGO; then
    info "Building Rust targets..."
    cd "$PROJECT_ROOT/src/rust"

    # Always build CPU crates
    cargo build --release -p mlp-cpu -p cnn-cpu 2>&1 | tail -5
    ok "Rust (CPU) — mlp-cpu, cnn-cpu"
    built+=("Rust (CPU)")

    # GPU crates only if CUDA available
    if $HAS_NVCC; then
        cargo build --release -p mlp-cuda-cublas -p mlp-cuda-kernels \
                              -p cnn-cuda-cublas -p cnn-cuda-kernels 2>&1 | tail -5
        ok "Rust (cuBLAS) — mlp-cuda-cublas, cnn-cuda-cublas"
        ok "Rust (CUDA Kernels) — mlp-cuda-kernels, cnn-cuda-kernels"
        built+=("Rust (cuBLAS)" "Rust (CUDA Kernels)")
    else
        skipped+=("Rust GPU crates — nvcc not found")
    fi
else
    skipped+=("Rust targets — cargo not found")
fi
cd "$PROJECT_ROOT"
echo ""

# ─────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════╗"
echo "║               Build Summary                  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
if [ ${#built[@]} -gt 0 ]; then
    for b in "${built[@]}"; do
        ok "$b"
    done
fi
if [ ${#skipped[@]} -gt 0 ]; then
    echo ""
    for s in "${skipped[@]}"; do
        warn "$s"
    done
fi
echo ""
info "Run benchmarks with:"
echo "  $PYTHON src/scripts/benchmark.py --mode scaling --runs 1"
echo "  $PYTHON src/scripts/benchmark.py --mode scaling --model cnn --runs 1"
echo ""
