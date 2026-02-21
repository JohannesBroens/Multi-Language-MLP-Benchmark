# Rust MLP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three Rust MLP implementations (CPU, cuBLAS GPU, raw CUDA kernels GPU) to the benchmark suite, matching the exact algorithm and output format of existing C/Python implementations.

**Architecture:** Cargo workspace with a shared `mlp-common` crate for data loading/CLI/output, plus three binary crates: `mlp-cpu` (Rayon + tiled GEMM), `mlp-cuda-cublas` (cuBLAS FFI), and `mlp-cuda-kernels` (custom .cu kernels via FFI). Each binary produces the same standardized output parsed by `benchmark.py`.

**Tech Stack:** Rust 1.93.0, Rayon (CPU parallelism), num_cpus (physical core detection), CUDA 13.1 + cuBLAS (via raw FFI bindings in build.rs — no crates.io CUDA crates needed for stability), nvcc for .cu compilation.

---

### Task 1: Scaffold Cargo Workspace + mlp-common Crate

**Files:**
- Create: `src/rust/Cargo.toml` (workspace root)
- Create: `src/rust/mlp-common/Cargo.toml`
- Create: `src/rust/mlp-common/src/lib.rs`

**Step 1: Create workspace Cargo.toml**

```toml
# src/rust/Cargo.toml
[workspace]
members = ["mlp-common", "mlp-cpu", "mlp-cuda-cublas", "mlp-cuda-kernels"]
resolver = "2"
```

**Step 2: Create mlp-common crate**

`src/rust/mlp-common/Cargo.toml`:
```toml
[package]
name = "mlp-common"
version = "0.1.0"
edition = "2021"
```

`src/rust/mlp-common/src/lib.rs` must implement:
- `struct Args` with clap-like manual parsing: `--dataset`, `--batch-size`, `--num-samples`, `--hidden-size`, `--epochs`
- `fn parse_args() -> Args` using `std::env::args`
- Data loaders mirroring C's `data_loader.c`:
  - `load_generated_data(num_samples) -> (Vec<f32>, Vec<i32>, usize, usize, usize)` — synthetic 2D circle
  - `load_iris_data() -> ...` — reads `data/iris_processed.txt` (comma-separated: 4 floats, 1 float label)
  - `load_wine_data(path) -> ...` — reads semicolon-delimited CSV with header (11 features, quality label)
  - `load_breast_cancer_data() -> ...` — reads `data/wdbc.data` (id,diagnosis,30 features)
  - `fn load_dataset(args: &Args) -> Dataset` that dispatches
- `struct Dataset { inputs: Vec<f32>, labels: Vec<i32>, num_samples: usize, input_size: usize, output_size: usize }`
- `fn normalize_features(inputs: &mut [f32], num_samples: usize, input_size: usize)` — z-score per feature (eps=1e-8)
- `fn shuffle_data(inputs: &mut [f32], labels: &mut [i32], num_samples: usize, input_size: usize)` — Fisher-Yates
- `fn train_test_split(dataset: &Dataset) -> (train_inputs, train_labels, test_inputs, test_labels)` — 80/20

Key matching details from C `data_loader.c`:
- Generated: points in [-1,1]^2, label 1 if x^2+y^2 < 0.25, else 0. Default 1000 samples.
- Iris: 4 features, float label rounded to int, 3 classes
- Wine: semicolon delim, skip header, 11 features, quality 0-10 as label, output_size=11
- Breast cancer: id(skip), diagnosis(M=1,B=0), 30 features, output_size=2

**Step 3: Verify build**

```bash
cd src/rust && cargo check -p mlp-common
```

Expected: compiles successfully

**Step 4: Commit**

```bash
git add src/rust/
git commit -m "feat(rust): scaffold cargo workspace and mlp-common data loading crate"
```

---

### Task 2: Implement Rust CPU MLP (Rayon + Tiled GEMM)

**Files:**
- Create: `src/rust/mlp-cpu/Cargo.toml`
- Create: `src/rust/mlp-cpu/src/main.rs`

**Step 1: Create mlp-cpu crate**

`src/rust/mlp-cpu/Cargo.toml`:
```toml
[package]
name = "mlp-cpu"
version = "0.1.0"
edition = "2021"

[dependencies]
mlp-common = { path = "../mlp-common" }
rayon = "1.10"
num_cpus = "1.16"
```

**Step 2: Implement main.rs**

The CPU implementation must match C's `mlp_cpu.c` algorithm exactly:

1. **Physical core detection**: Use `num_cpus::get_physical()` to set rayon threadpool to physical cores only (avoids HT contention on FPU-bound GEMM).

2. **GEMM functions** (all row-major, f32):
   - `sgemm_nn(M, N, K, A, B, C)` — C[M,N] = A[M,K] @ B[K,N], cache-tiled i-k-j with TILE=64
   - `sgemm_tn(M, N, K, A, B, C)` — C[M,N] = A^T[M,K] @ B[K,N], A stored as [K,M]
   - `sgemm_nt(M, N, K, A, B, C)` — C[M,N] = A[M,K] @ B^T[K,N], B stored as [N,K]
   - Threshold guards: only use rayon parallel iter if `M*N*K > 100_000` (GEMM) or `len > 200_000` (elementwise)

3. **MLP struct**: `input_weights: Vec<f32>`, `output_weights: Vec<f32>`, `hidden_biases: Vec<f32>`, `output_biases: Vec<f32>`, plus dimension fields.

4. **Xavier init**: `Uniform(-sqrt(2/fan_in), sqrt(2/fan_in))` using a simple xorshift32 PRNG (matches C).

5. **Training loop** (mirrors C `mlp_train`):
   - For each epoch, for each mini-batch:
     - Forward: `hidden = ReLU(X @ W1 + b1)`, `output = softmax(hidden @ W2 + b2)`
     - Loss: cross-entropy `= -sum(log(output[target] + 1e-7))`
     - Backward: `d_output = output - one_hot`, `grad_W2 = hidden^T @ d_output`, `d_hidden = (d_output @ W2^T) * relu_mask`, `grad_W1 = X^T @ d_hidden`
     - SGD: `param -= (lr / batch_size) * grad`
   - Print epoch loss every 100 epochs

6. **Evaluation**: forward pass, compute avg loss + accuracy %

7. **main()**: parse args, load dataset, normalize, shuffle, split 80/20, train, evaluate, print standardized output:
   ```
   === Results on <dataset> ===
   Test Loss:     0.1234
   Test Accuracy: 96.67%
   Train time:    0.123 s
   Eval time:     0.001 s
   Throughput:    1234567 samples/s
   ```
   Use `std::time::Instant` for CLOCK_MONOTONIC-equivalent timing.

**Step 3: Build and test**

```bash
cd src/rust && cargo build --release -p mlp-cpu
../../src/rust/target/release/mlp-cpu --dataset generated --num-samples 10000 --epochs 100 --hidden-size 64 --batch-size 512
```

Expected: prints standardized output with reasonable accuracy (>80%) and throughput.

**Step 4: Cross-validate against C CPU**

```bash
# Both should produce similar accuracy on same dataset
./src/c/build_cpu/main --dataset iris --epochs 1000
./src/rust/target/release/mlp-cpu --dataset iris --epochs 1000
```

Expected: Both show ~95-97% accuracy on iris.

**Step 5: Commit**

```bash
git add src/rust/mlp-cpu/
git commit -m "feat(rust): add CPU MLP with Rayon parallelism and tiled GEMM"
```

---

### Task 3: Implement Rust cuBLAS GPU MLP

**Files:**
- Create: `src/rust/mlp-cuda-cublas/Cargo.toml`
- Create: `src/rust/mlp-cuda-cublas/build.rs`
- Create: `src/rust/mlp-cuda-cublas/src/main.rs`

**Step 1: Create crate with CUDA FFI bindings**

`src/rust/mlp-cuda-cublas/Cargo.toml`:
```toml
[package]
name = "mlp-cuda-cublas"
version = "0.1.0"
edition = "2021"
build = "build.rs"

[dependencies]
mlp-common = { path = "../mlp-common" }

[build-dependencies]
# None needed — we link CUDA manually
```

`src/rust/mlp-cuda-cublas/build.rs` — link against CUDA runtime + cuBLAS:
```rust
fn main() {
    // Find CUDA — try CUDA_PATH env, then /usr/local/cuda
    let cuda_path = std::env::var("CUDA_PATH")
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:rustc-link-search=native={cuda_path}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
}
```

**Step 2: Implement CUDA FFI bindings + MLP**

`src/rust/mlp-cuda-cublas/src/main.rs` must implement:

1. **Raw FFI declarations** (extern "C") for:
   - `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemset`, `cudaDeviceSynchronize`
   - `cublasCreate_v2`, `cublasDestroy_v2`, `cublasSgemm_v2`
   - Use `repr(C)` enums for `cublasOperation_t`, `cudaMemcpyKind`
   - Handle type: `type cublasHandle_t = *mut std::ffi::c_void`

2. **GEMM wrappers** (row-major via transposition trick, matching C `mlp.cu`):
   - `sgemm_nn(handle, M, N, K, A, B, C)` — B_col @ A_col = C_col (swap A,B + swap M,N)
   - `sgemm_tn(handle, M, N, K, A, B, C)` — B_col @ A^T_col
   - `sgemm_nt(handle, M, N, K, A, B, C)` — B^T_col @ A_col

3. **Element-wise CUDA kernels** — since we can't write __global__ in Rust, we need a small .cu helper. BUT for the cuBLAS variant, we do element-wise ops ON THE HOST after copying back, OR we write a tiny .cu file with the helper kernels.

   **Decision: Write a small `kernels.cu` with element-wise kernels, compiled by build.rs.**

   Update build.rs to also compile `src/kernels.cu` via `cc` crate:
   ```rust
   // build.rs addition:
   cc::Build::new()
       .cuda(true)
       .file("src/kernels.cu")
       .compile("mlp_cublas_kernels");
   ```

   `src/rust/mlp-cuda-cublas/src/kernels.cu`:
   - `extern "C" void launch_bias_relu(float *hidden, const float *bias, int total, int hid)`
   - `extern "C" void launch_bias_softmax(float *output, const float *bias, int bs, int out)`
   - `extern "C" void launch_ce_grad(const float *output, const int *targets, float *d_output, float *losses, int bs, int out)`
   - `extern "C" void launch_relu_mask(float *d_hidden, const float *hidden, int total)`
   - `extern "C" void launch_col_sum(const float *mat, float *out_vec, int rows, int cols)`
   - `extern "C" void launch_sgd(float *param, const float *grad, float lr, int n)`
   - `extern "C" void launch_reduce_sum(const float *arr, float *result, int n)`
   - `extern "C" void launch_init_weights(float *weights, int size, float scale, unsigned long seed)`

   These are thin C wrappers around the __global__ kernels.

4. **Training loop**: Mirrors C `mlp.cu` exactly — all data on device, cuBLAS for GEMM, kernels for elementwise.

5. **Main**: Same flow as CPU — parse args, load on host, copy to device, train, evaluate, print standardized output.

**Step 3: Build and test**

```bash
cd src/rust && cargo build --release -p mlp-cuda-cublas
./src/rust/target/release/mlp-cuda-cublas --dataset generated --num-samples 10000 --epochs 100
```

**Step 4: Commit**

```bash
git add src/rust/mlp-cuda-cublas/
git commit -m "feat(rust): add cuBLAS GPU MLP via CUDA FFI"
```

---

### Task 4: Implement Rust Raw CUDA Kernels GPU MLP

**Files:**
- Create: `src/rust/mlp-cuda-kernels/Cargo.toml`
- Create: `src/rust/mlp-cuda-kernels/build.rs`
- Create: `src/rust/mlp-cuda-kernels/src/kernels.cu`
- Create: `src/rust/mlp-cuda-kernels/src/main.rs`

**Step 1: Create crate**

Same structure as cuBLAS but WITHOUT cuBLAS dependency — only `cudart`.

`build.rs`:
```rust
fn main() {
    let cuda_path = std::env::var("CUDA_PATH")
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:rustc-link-search=native={cuda_path}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");

    cc::Build::new()
        .cuda(true)
        .flag("-O3")
        .flag("-arch=native")  // auto-detect GPU architecture
        .file("src/kernels.cu")
        .compile("mlp_raw_kernels");
}
```

**Step 2: Write custom CUDA kernels**

`src/rust/mlp-cuda-kernels/src/kernels.cu`:

Custom tiled matmul kernel (shared memory, 16x16 tiles):
```cuda
#define TILE_DIM 16
__global__ void matmul_nn(const float *A, const float *B, float *C,
                          int M, int N, int K) {
    // Shared memory tiled GEMM
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];
    // ... standard tiled implementation
}
```

Plus all elementwise kernels (same as Task 3 but matmul replaces cuBLAS).

All exposed via `extern "C" void launch_*()` wrappers.

**Step 3: Implement main.rs**

Same structure as cuBLAS variant but calls `launch_matmul_nn` etc. instead of cuBLAS.

**Step 4: Build and test**

```bash
cd src/rust && cargo build --release -p mlp-cuda-kernels
./src/rust/target/release/mlp-cuda-kernels --dataset generated --num-samples 10000 --epochs 100
```

**Step 5: Commit**

```bash
git add src/rust/mlp-cuda-kernels/
git commit -m "feat(rust): add raw CUDA kernels GPU MLP"
```

---

### Task 5: Integrate into benchmark.py

**Files:**
- Modify: `src/scripts/benchmark.py`

**Step 1: Add Rust implementations to IMPLEMENTATIONS list**

After the PyTorch entries, add:
```python
("Rust (CPU)",
 "./src/rust/target/release/mlp-cpu --dataset {dataset} --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs}"),
("Rust (cuBLAS)",
 "./src/rust/target/release/mlp-cuda-cublas --dataset {dataset} --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs}"),
("Rust (CUDA Kernels)",
 "./src/rust/target/release/mlp-cuda-kernels --dataset {dataset} --batch-size {batch_size} --num-samples {num_samples} --hidden-size {hidden_size} --epochs {epochs}"),
```

**Step 2: Add is_available() checks**

```python
if label == "Rust (CPU)":
    return os.path.isfile(os.path.join(PROJECT_ROOT, "src", "rust", "target", "release", "mlp-cpu"))
if label == "Rust (cuBLAS)":
    return os.path.isfile(os.path.join(PROJECT_ROOT, "src", "rust", "target", "release", "mlp-cuda-cublas"))
if label == "Rust (CUDA Kernels)":
    return os.path.isfile(os.path.join(PROJECT_ROOT, "src", "rust", "target", "release", "mlp-cuda-kernels"))
```

**Step 3: Add plot styles**

```python
IMPL_STYLES["Rust (CPU)"]          = {"color": "#f39c12", "marker": "P", "linewidth": 2.2, "markersize": 8, "zorder": 5}
IMPL_STYLES["Rust (cuBLAS)"]       = {"color": "#1abc9c", "marker": "H", "linewidth": 2.8, "markersize": 9, "zorder": 9}
IMPL_STYLES["Rust (CUDA Kernels)"] = {"color": "#e91e63", "marker": "X", "linewidth": 2.8, "markersize": 9, "zorder": 8}
```

**Step 4: Add speedup pairs**

```python
("Rust (cuBLAS)", "Rust (CPU)", "#1abc9c", "Rust cuBLAS / Rust CPU"),
("Rust (CUDA Kernels)", "Rust (CPU)", "#e91e63", "Rust Kernels / Rust CPU"),
("Rust (cuBLAS)", "C (CUDA)", "#f39c12", "Rust cuBLAS / C CUDA"),
```

**Step 5: Test benchmark with Rust**

```bash
.venv/bin/python src/scripts/benchmark.py --mode standard --datasets generated --runs 1
```

Expected: Rust implementations appear in output and produce valid results.

**Step 6: Commit**

```bash
git add src/scripts/benchmark.py
git commit -m "feat(benchmark): integrate Rust CPU, cuBLAS, and CUDA kernels implementations"
```

---

### Task 6: Update CLAUDE.md and Build Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add Rust build and run commands**

Add to Build Commands section:
```markdown
### Rust (all targets)
\```bash
cd src/rust && cargo build --release
\```

### Rust (CPU only, no CUDA needed)
\```bash
cd src/rust && cargo build --release -p mlp-cpu
\```

### Run Rust Implementations
\```bash
# Rust CPU
./src/rust/target/release/mlp-cpu --dataset <name>

# Rust cuBLAS
./src/rust/target/release/mlp-cuda-cublas --dataset <name>

# Rust CUDA Kernels
./src/rust/target/release/mlp-cuda-kernels --dataset <name>
\```
```

**Step 2: Update Architecture section**

Add Rust implementation description.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add Rust build/run instructions to CLAUDE.md"
```

---

### Task 7: Run Full Scaling Benchmark with Rust

**Step 1: Build everything**

```bash
cd src/rust && cargo build --release
```

**Step 2: Quick smoke test all 3 Rust binaries**

```bash
for bin in mlp-cpu mlp-cuda-cublas mlp-cuda-kernels; do
    echo "=== $bin ===" && ./src/rust/target/release/$bin --dataset generated --num-samples 5000 --epochs 50 --batch-size 256
done
```

**Step 3: Run scaling benchmark**

```bash
.venv/bin/python src/scripts/benchmark.py --mode scaling --scaling-epochs 100 --runs 1
```

**Step 4: Review generated plots**

All 7+ plots in `figs/` should now include 8 implementations with Rust entries visible.
