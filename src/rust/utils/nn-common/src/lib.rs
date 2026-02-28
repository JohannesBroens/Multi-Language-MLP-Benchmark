use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::time::{SystemTime, UNIX_EPOCH};

use rayon::prelude::*;

// Re-export for binary crates
pub use rayon;
pub use num_cpus;

// ---------------------------------------------------------------------------
// PRNG — xorshift32
// ---------------------------------------------------------------------------

pub struct Xorshift32 {
    state: u32,
}

impl Xorshift32 {
    pub fn new(seed: u32) -> Self {
        // Avoid zero state (xorshift32 fixpoint)
        Xorshift32 {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Returns a value in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32 + 1.0)
    }
}

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------

pub struct Args {
    pub dataset: String,
    pub batch_size: usize,
    pub num_samples: usize, // 0 = use dataset default
    pub hidden_size: usize,
    pub epochs: usize,
    pub learning_rate: f32,  // 0.0 = use binary's default
    pub optimizer: String,   // "sgd" or "adam"
    pub scheduler: String,   // "none" or "cosine"
}

fn print_usage(prog: &str) {
    eprintln!(
        "Usage: {} --dataset [generated|iris|wine-red|wine-white|breast-cancer|mnist]",
        prog
    );
    eprintln!("  --batch-size N    Mini-batch size (default: 4096)");
    eprintln!("  --num-samples N   Number of samples for generated dataset (default: 1000)");
    eprintln!("  --hidden-size N   Hidden layer size (default: 64)");
    eprintln!("  --epochs N        Number of training epochs (default: 1000)");
    eprintln!("  --learning-rate F Learning rate (default: model-specific)");
    eprintln!("  --optimizer STR   Optimizer: sgd|adam (default: sgd)");
    eprintln!("  --scheduler STR   LR scheduler: none|cosine (default: none)");
}

pub fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let prog = argv.first().map(|s| s.as_str()).unwrap_or("mlp");

    let mut dataset: Option<String> = None;
    let mut batch_size: usize = 4096;
    let mut num_samples: usize = 0;
    let mut hidden_size: usize = 64;
    let mut epochs: usize = 1000;
    let mut learning_rate: f32 = 0.0;
    let mut optimizer: String = "sgd".to_string();
    let mut scheduler: String = "none".to_string();

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--dataset" | "-d" => {
                i += 1;
                if i >= argv.len() {
                    print_usage(prog);
                    std::process::exit(1);
                }
                dataset = Some(argv[i].clone());
            }
            "--batch-size" => {
                i += 1;
                if i >= argv.len() {
                    print_usage(prog);
                    std::process::exit(1);
                }
                batch_size = argv[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --batch-size value: {}", argv[i]);
                    std::process::exit(1);
                });
            }
            "--num-samples" => {
                i += 1;
                if i >= argv.len() {
                    print_usage(prog);
                    std::process::exit(1);
                }
                num_samples = argv[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --num-samples value: {}", argv[i]);
                    std::process::exit(1);
                });
            }
            "--hidden-size" => {
                i += 1;
                if i >= argv.len() {
                    print_usage(prog);
                    std::process::exit(1);
                }
                hidden_size = argv[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --hidden-size value: {}", argv[i]);
                    std::process::exit(1);
                });
            }
            "--epochs" => {
                i += 1;
                if i >= argv.len() {
                    print_usage(prog);
                    std::process::exit(1);
                }
                epochs = argv[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --epochs value: {}", argv[i]);
                    std::process::exit(1);
                });
            }
            "--learning-rate" => {
                i += 1;
                if i >= argv.len() {
                    print_usage(prog);
                    std::process::exit(1);
                }
                learning_rate = argv[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --learning-rate value: {}", argv[i]);
                    std::process::exit(1);
                });
            }
            "--optimizer" => {
                i += 1;
                if i >= argv.len() {
                    print_usage(prog);
                    std::process::exit(1);
                }
                optimizer = argv[i].clone();
            }
            "--scheduler" => {
                i += 1;
                if i >= argv.len() {
                    print_usage(prog);
                    std::process::exit(1);
                }
                scheduler = argv[i].clone();
            }
            _ => {
                print_usage(prog);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if dataset.is_none() {
        print_usage(prog);
        std::process::exit(1);
    }

    Args {
        dataset: dataset.unwrap(),
        batch_size,
        num_samples,
        hidden_size,
        epochs,
        learning_rate,
        optimizer,
        scheduler,
    }
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

pub struct Dataset {
    pub inputs: Vec<f32>,  // [num_samples * input_size] row-major
    pub labels: Vec<i32>,  // [num_samples]
    pub num_samples: usize,
    pub input_size: usize,
    pub output_size: usize, // number of classes
}

// ---------------------------------------------------------------------------
// Data loaders — match C data_loader.c exactly
// ---------------------------------------------------------------------------

/// Generate synthetic 2D circle classification data.
/// Points in [-1, 1]^2; label 1 if inside circle of radius 0.5, else 0.
pub fn load_generated_data(num_samples_requested: usize) -> Dataset {
    let num_samples = if num_samples_requested > 0 {
        num_samples_requested
    } else {
        1000
    };
    let input_size = 2;
    let output_size = 2;

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as u32;
    let mut rng = Xorshift32::new(seed);

    let mut inputs = vec![0.0f32; num_samples * input_size];
    let mut labels = vec![0i32; num_samples];

    for i in 0..num_samples {
        let x = rng.next_f32() * 2.0 - 1.0;
        let y = rng.next_f32() * 2.0 - 1.0;
        inputs[i * input_size] = x;
        inputs[i * input_size + 1] = y;
        labels[i] = if x * x + y * y < 0.25 { 1 } else { 0 };
    }

    Dataset {
        inputs,
        labels,
        num_samples,
        input_size,
        output_size,
    }
}

/// Load Iris dataset from preprocessed file.
/// Format: comma-separated, 4 float features then float label per line.
/// File: data/iris_processed.txt (150 samples, 3 classes)
pub fn load_iris_data() -> Dataset {
    let filename = "data/iris_processed.txt";
    let input_size = 4;
    let output_size = 3;

    let file = File::open(filename).unwrap_or_else(|e| {
        eprintln!("Failed to open Iris dataset file '{}': {}", filename, e);
        std::process::exit(1);
    });
    let reader = BufReader::new(file);

    let mut inputs = Vec::new();
    let mut labels = Vec::new();
    let mut num_samples = 0;

    for line in reader.lines() {
        let line = line.unwrap_or_else(|e| {
            eprintln!("Error reading {}: {}", filename, e);
            std::process::exit(1);
        });
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        // 4 features + 1 label
        for j in 0..input_size {
            let val: f32 = parts[j].trim().parse().unwrap_or_else(|e| {
                eprintln!("Parse error in {}: {}", filename, e);
                std::process::exit(1);
            });
            inputs.push(val);
        }
        let label_f: f32 = parts[input_size].trim().parse().unwrap_or_else(|e| {
            eprintln!("Parse error in {}: {}", filename, e);
            std::process::exit(1);
        });
        labels.push((label_f + 0.5) as i32);
        num_samples += 1;
    }

    Dataset {
        inputs,
        labels,
        num_samples,
        input_size,
        output_size,
    }
}

/// Load UCI Wine Quality dataset.
/// Format: semicolon-delimited CSV with header row.
/// 11 float features; last column is quality score (0-10) used as label.
pub fn load_wine_data(filename: &str) -> Dataset {
    let input_size = 11;
    let output_size = 11; // quality scores 0-10

    let file = File::open(filename).unwrap_or_else(|e| {
        eprintln!(
            "Failed to open Wine Quality dataset file '{}': {}",
            filename, e
        );
        std::process::exit(1);
    });
    let reader = BufReader::new(file);

    let mut inputs = Vec::new();
    let mut labels = Vec::new();
    let mut num_samples = 0;
    let mut first_line = true;

    for line in reader.lines() {
        let line = line.unwrap_or_else(|e| {
            eprintln!("Error reading {}: {}", filename, e);
            std::process::exit(1);
        });
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Skip header line
        if first_line {
            first_line = false;
            continue;
        }

        let parts: Vec<&str> = line.split(';').collect();
        for j in 0..input_size {
            let val: f32 = parts[j].trim().parse().unwrap_or_else(|e| {
                eprintln!("Parse error in {}: {}", filename, e);
                std::process::exit(1);
            });
            inputs.push(val);
        }
        let label: i32 = parts[input_size].trim().parse().unwrap_or_else(|e| {
            eprintln!("Parse error in {}: {}", filename, e);
            std::process::exit(1);
        });
        labels.push(label);
        num_samples += 1;
    }

    Dataset {
        inputs,
        labels,
        num_samples,
        input_size,
        output_size,
    }
}

/// Load Wisconsin Breast Cancer Diagnostic dataset.
/// Format: id,diagnosis(M/B),30 float features -- comma-separated, no header.
/// M (malignant) = 1, B (benign) = 0.
/// File: data/wdbc.data (569 samples, 2 classes)
pub fn load_breast_cancer_data() -> Dataset {
    let filename = "data/wdbc.data";
    let input_size = 30;
    let output_size = 2;

    let file = File::open(filename).unwrap_or_else(|e| {
        eprintln!(
            "Failed to open Breast Cancer dataset file '{}': {}",
            filename, e
        );
        std::process::exit(1);
    });
    let reader = BufReader::new(file);

    let mut inputs = Vec::new();
    let mut labels = Vec::new();
    let mut num_samples = 0;

    for line in reader.lines() {
        let line = line.unwrap_or_else(|e| {
            eprintln!("Error reading {}: {}", filename, e);
            std::process::exit(1);
        });
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        // parts[0] = id (discard), parts[1] = diagnosis, parts[2..32] = features
        let diagnosis = parts[1].trim();
        labels.push(if diagnosis == "M" { 1 } else { 0 });

        for j in 0..input_size {
            let val: f32 = parts[2 + j].trim().parse().unwrap_or_else(|e| {
                eprintln!("Parse error in {}: {}", filename, e);
                std::process::exit(1);
            });
            inputs.push(val);
        }
        num_samples += 1;
    }

    Dataset {
        inputs,
        labels,
        num_samples,
        input_size,
        output_size,
    }
}

// ---------------------------------------------------------------------------
// Processing functions
// ---------------------------------------------------------------------------

/// Z-score normalization per feature: x' = (x - mean) / sqrt(var/n + 1e-8).
/// Uses population variance (divides by n, not n-1) to match C version.
pub fn normalize_features(inputs: &mut [f32], num_samples: usize, input_size: usize) {
    for j in 0..input_size {
        let mut mean = 0.0f32;
        for i in 0..num_samples {
            mean += inputs[i * input_size + j];
        }
        mean /= num_samples as f32;

        let mut var = 0.0f32;
        for i in 0..num_samples {
            let d = inputs[i * input_size + j] - mean;
            var += d * d;
        }
        let std = (var / num_samples as f32 + 1e-8f32).sqrt();

        for i in 0..num_samples {
            inputs[i * input_size + j] = (inputs[i * input_size + j] - mean) / std;
        }
    }
}

/// Fisher-Yates shuffle: randomly permute samples in-place.
/// Swaps both input feature rows and corresponding labels.
pub fn shuffle_data(
    inputs: &mut [f32],
    labels: &mut [i32],
    num_samples: usize,
    input_size: usize,
) {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u32;
    let mut rng = Xorshift32::new(seed);

    for i in (1..num_samples).rev() {
        let j = (rng.next_u32() as usize) % (i + 1);
        // Swap labels
        labels.swap(i, j);
        // Swap input rows
        for k in 0..input_size {
            inputs.swap(i * input_size + k, j * input_size + k);
        }
    }
}

/// Read big-endian u32 from a reader.
fn read_be_u32(reader: &mut impl Read) -> u32 {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).expect("Failed to read IDX header");
    u32::from_be_bytes(buf)
}

/// Load MNIST handwritten digits (IDX binary format).
/// Pixel values normalized to [0, 1].
/// If is_test is true, loads the 10K test set instead of the 60K training set.
pub fn load_mnist_data(is_test: bool) -> Dataset {
    let img_path = if is_test { "data/t10k-images-idx3-ubyte" } else { "data/train-images-idx3-ubyte" };
    let lbl_path = if is_test { "data/t10k-labels-idx1-ubyte" } else { "data/train-labels-idx1-ubyte" };

    // Read images
    let mut fimg = File::open(img_path).unwrap_or_else(|e| {
        eprintln!("Cannot open {}: {}", img_path, e);
        std::process::exit(1);
    });

    let magic = read_be_u32(&mut fimg);
    assert_eq!(magic, 2051, "Invalid MNIST image file magic number");

    let n = read_be_u32(&mut fimg) as usize;
    let rows = read_be_u32(&mut fimg) as usize;
    let cols = read_be_u32(&mut fimg) as usize;
    let pixels = rows * cols;

    let mut img_buf = vec![0u8; n * pixels];
    fimg.read_exact(&mut img_buf).expect("Failed to read MNIST image data");

    let inputs: Vec<f32> = img_buf.iter().map(|&b| b as f32 / 255.0).collect();

    // Read labels
    let mut flbl = File::open(lbl_path).unwrap_or_else(|e| {
        eprintln!("Cannot open {}: {}", lbl_path, e);
        std::process::exit(1);
    });

    let _magic = read_be_u32(&mut flbl);
    let _count = read_be_u32(&mut flbl);

    let mut lbl_buf = vec![0u8; n];
    flbl.read_exact(&mut lbl_buf).expect("Failed to read MNIST label data");

    let labels: Vec<i32> = lbl_buf.iter().map(|&b| b as i32).collect();

    Dataset {
        inputs,
        labels,
        num_samples: n,
        input_size: pixels,
        output_size: 10,
    }
}

/// Dispatch to the right loader based on dataset name.
pub fn load_dataset(args: &Args) -> Dataset {
    match args.dataset.as_str() {
        "generated" => load_generated_data(args.num_samples),
        "iris" => load_iris_data(),
        "wine-red" => load_wine_data("data/winequality-red.csv"),
        "wine-white" => load_wine_data("data/winequality-white.csv"),
        "breast-cancer" => load_breast_cancer_data(),
        "mnist" => load_mnist_data(false),
        other => {
            eprintln!("Unknown dataset: {}", other);
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
//  Constants -- match C nn_ops.h exactly
// ---------------------------------------------------------------------------

/// Tile size for L1 cache residency (~32KB): 3 tiles of 64x64 floats = 48KB.
pub const TILE: usize = 64;

/// Minimum FLOPs (M*N*K) to justify rayon parallel overhead.
pub const OMP_FLOP_THRESHOLD: i64 = 100_000;

/// Minimum elements for elementwise parallel (bias+relu, softmax, SGD).
pub const OMP_ELEM_THRESHOLD: i64 = 200_000;

// ---------------------------------------------------------------------------
//  Thread pool initialization
// ---------------------------------------------------------------------------

/// Initialize the global rayon thread pool to use physical cores only.
/// Should be called once at program startup.
pub fn init_thread_pool() {
    let physical_cores = num_cpus::get_physical();
    rayon::ThreadPoolBuilder::new()
        .num_threads(physical_cores)
        .build_global()
        .unwrap();
}

// ---------------------------------------------------------------------------
//  GEMM -- cache-tiled, i-k-j order (matches C nn_sgemm_nn/tn/nt)
// ---------------------------------------------------------------------------

/// C[M,N] = A[M,K] @ B[K,N] -- cache-friendly i-k-j with optional tiling.
pub fn sgemm_nn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let flops = m as i64 * n as i64 * k as i64;
    let c = &mut c[..m * n];

    if k <= TILE && n <= TILE {
        if flops > OMP_FLOP_THRESHOLD {
            c.par_chunks_mut(n).enumerate().for_each(|(i, ci)| {
                for j in 0..n { ci[j] = 0.0; }
                for kk in 0..k {
                    let a_ik = a[i * k + kk];
                    let bk = &b[kk * n..kk * n + n];
                    for j in 0..n { ci[j] += a_ik * bk[j]; }
                }
            });
        } else {
            for i in 0..m {
                let ci = &mut c[i * n..(i + 1) * n];
                let ai = &a[i * k..];
                for j in 0..n { ci[j] = 0.0; }
                for kk in 0..k {
                    let a_ik = ai[kk];
                    let bk = &b[kk * n..kk * n + n];
                    for j in 0..n { ci[j] += a_ik * bk[j]; }
                }
            }
        }
    } else {
        c[..m * n].fill(0.0);

        if flops > OMP_FLOP_THRESHOLD {
            // Split output rows into TILE-sized chunks for parallel processing.
            // Each chunk gets exclusive &mut access, enabling autovectorization
            // (equivalent to C's restrict pointer guarantee).
            c[..m * n].par_chunks_mut(TILE * n).enumerate().for_each(|(it, c_chunk)| {
                let i0 = it * TILE;
                let imax = (i0 + TILE).min(m);
                let chunk_rows = imax - i0;
                for k0 in (0..k).step_by(TILE) {
                    let kmax = (k0 + TILE).min(k);
                    for j0 in (0..n).step_by(TILE) {
                        let jmax = (j0 + TILE).min(n);
                        for ii in 0..chunk_rows {
                            let ci = &mut c_chunk[ii * n..];
                            for kk in k0..kmax {
                                let a_ik = a[(i0 + ii) * k + kk];
                                let bk = &b[kk * n..];
                                for j in j0..jmax {
                                    ci[j] += a_ik * bk[j];
                                }
                            }
                        }
                    }
                }
            });
        } else {
            for i0 in (0..m).step_by(TILE) {
                let imax = (i0 + TILE).min(m);
                for k0 in (0..k).step_by(TILE) {
                    let kmax = (k0 + TILE).min(k);
                    for j0 in (0..n).step_by(TILE) {
                        let jmax = (j0 + TILE).min(n);
                        for i in i0..imax {
                            for kk in k0..kmax {
                                let a_ik = a[i * k + kk];
                                for j in j0..jmax {
                                    c[i * n + j] += a_ik * b[kk * n + j];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// C[M,N] = A^T[M,K] @ B[K,N], A stored as [K,M].
pub fn sgemm_tn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let flops = m as i64 * n as i64 * k as i64;
    let c = &mut c[..m * n];

    if flops > OMP_FLOP_THRESHOLD {
        c.par_chunks_mut(n).enumerate().for_each(|(i, ci)| {
            for j in 0..n { ci[j] = 0.0; }
            for kk in 0..k {
                let a_ki = a[kk * m + i];
                let bk = &b[kk * n..kk * n + n];
                for j in 0..n { ci[j] += a_ki * bk[j]; }
            }
        });
    } else {
        for i in 0..m {
            let ci = &mut c[i * n..(i + 1) * n];
            for j in 0..n { ci[j] = 0.0; }
            for kk in 0..k {
                let a_ki = a[kk * m + i];
                let bk = &b[kk * n..kk * n + n];
                for j in 0..n { ci[j] += a_ki * bk[j]; }
            }
        }
    }
}

/// C[M,N] = A[M,K] @ B^T[K,N], B stored as [N,K].
pub fn sgemm_nt(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let flops = m as i64 * n as i64 * k as i64;
    let c = &mut c[..m * n];

    if flops > OMP_FLOP_THRESHOLD {
        c.par_chunks_mut(n).enumerate().for_each(|(i, ci)| {
            let ai = &a[i * k..i * k + k];
            for j in 0..n {
                let mut sum = 0.0f32;
                let bj = &b[j * k..j * k + k];
                for kk in 0..k { sum += ai[kk] * bj[kk]; }
                ci[j] = sum;
            }
        });
    } else {
        for i in 0..m {
            let ai = &a[i * k..i * k + k];
            for j in 0..n {
                let mut sum = 0.0f32;
                let bj = &b[j * k..j * k + k];
                for kk in 0..k { sum += ai[kk] * bj[kk]; }
                c[i * n + j] = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  Activation functions
// ---------------------------------------------------------------------------

/// In-place ReLU: x[i] = max(0, x[i])
pub fn relu_forward(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 { *v = 0.0; }
    }
}

/// In-place ReLU backward: grad[i] = (x[i] > 0) ? grad[i] : 0
pub fn relu_backward(grad: &mut [f32], x: &[f32]) {
    for (g, &xi) in grad.iter_mut().zip(x.iter()) {
        if xi <= 0.0 { *g = 0.0; }
    }
}

/// In-place bias + ReLU fused for a [rows, width] matrix
pub fn bias_relu(x: &mut [f32], bias: &[f32], rows: usize, width: usize) {
    let n = rows as i64 * width as i64;
    if n > OMP_ELEM_THRESHOLD {
        x.par_chunks_mut(width).for_each(|row| {
            for j in 0..width {
                row[j] += bias[j];
                if row[j] < 0.0 { row[j] = 0.0; }
            }
        });
    } else {
        for i in 0..rows {
            let row = &mut x[i * width..(i + 1) * width];
            for j in 0..width {
                row[j] += bias[j];
                if row[j] < 0.0 { row[j] = 0.0; }
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  Softmax + Cross-Entropy
// ---------------------------------------------------------------------------

/// In-place numerically stable softmax per row.
pub fn softmax(x: &mut [f32], rows: usize, cols: usize) {
    let n = rows as i64 * cols as i64;
    if n > OMP_ELEM_THRESHOLD {
        x.par_chunks_mut(cols).for_each(|row| {
            let max_val = row.iter().cloned().fold(-1e30f32, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            for v in row.iter_mut() { *v /= sum; }
        });
    } else {
        for i in 0..rows {
            let row = &mut x[i * cols..(i + 1) * cols];
            let max_val = row.iter().cloned().fold(-1e30f32, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            for v in row.iter_mut() { *v /= sum; }
        }
    }
}

/// In-place bias + softmax fused
pub fn bias_softmax(x: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    let n = rows as i64 * cols as i64;
    if n > OMP_ELEM_THRESHOLD {
        x.par_chunks_mut(cols).for_each(|row| {
            let mut max_val = -1e30f32;
            for j in 0..cols {
                row[j] += bias[j];
                if row[j] > max_val { max_val = row[j]; }
            }
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            for v in row.iter_mut() { *v /= sum; }
        });
    } else {
        for i in 0..rows {
            let row = &mut x[i * cols..(i + 1) * cols];
            let mut max_val = -1e30f32;
            for j in 0..cols {
                row[j] += bias[j];
                if row[j] > max_val { max_val = row[j]; }
            }
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            for v in row.iter_mut() { *v /= sum; }
        }
    }
}

/// Cross-entropy loss: returns average -log(prob[target]) over rows
pub fn cross_entropy_loss(probs: &[f32], targets: &[i32], rows: usize, cols: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..rows {
        total -= (probs[i * cols + targets[i] as usize] + 1e-7).ln();
    }
    total / rows as f32
}

/// Cross-entropy gradient: grad = probs - one_hot(targets).
/// Returns the total loss (sum, not average).
pub fn cross_entropy_grad(probs: &[f32], targets: &[i32], grad: &mut [f32], rows: usize, cols: usize) -> f32 {
    let mut total_loss = 0.0f32;
    for i in 0..rows {
        let p = &probs[i * cols..(i + 1) * cols];
        let g = &mut grad[i * cols..(i + 1) * cols];
        total_loss -= (p[targets[i] as usize] + 1e-7).ln();
        for j in 0..cols {
            g[j] = p[j] - if targets[i] == j as i32 { 1.0 } else { 0.0 };
        }
    }
    total_loss
}

/// Column sum: out[j] = sum_i(mat[i*cols+j])
pub fn col_sum(mat: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for v in out.iter_mut() { *v = 0.0; }
    for i in 0..rows {
        let row = &mat[i * cols..(i + 1) * cols];
        for j in 0..cols {
            out[j] += row[j];
        }
    }
}

/// SGD update: param[i] -= lr * grad[i]
pub fn sgd_update(param: &mut [f32], grad: &[f32], lr: f32) {
    let n = param.len() as i64;
    if n > OMP_ELEM_THRESHOLD {
        param.par_iter_mut().zip(grad.par_iter()).for_each(|(p, &g)| {
            *p -= lr * g;
        });
    } else {
        for (p, &g) in param.iter_mut().zip(grad.iter()) {
            *p -= lr * g;
        }
    }
}

/// Xavier uniform initialization: weights ~ Uniform(-scale, scale)
/// where scale = sqrt(2 / fan_in)
pub fn xavier_init(weights: &mut [f32], fan_in: usize, rng: &mut Xorshift32) {
    let scale = (2.0f32 / fan_in as f32).sqrt();
    for w in weights.iter_mut() {
        let r = (rng.next_u32() & 0x7FFF_FFFF) as f32 / 0x7FFF_FFFF_u32 as f32;
        *w = (r * 2.0 - 1.0) * scale;
    }
}
