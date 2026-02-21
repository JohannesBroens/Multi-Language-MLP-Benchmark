use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::{SystemTime, UNIX_EPOCH};

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
}

fn print_usage(prog: &str) {
    eprintln!(
        "Usage: {} --dataset [generated|iris|wine-red|wine-white|breast-cancer]",
        prog
    );
    eprintln!("  --batch-size N    Mini-batch size (default: 32)");
    eprintln!("  --num-samples N   Number of samples for generated dataset (default: 1000)");
    eprintln!("  --hidden-size N   Hidden layer size (default: 64)");
    eprintln!("  --epochs N        Number of training epochs (default: 1000)");
}

pub fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let prog = argv.first().map(|s| s.as_str()).unwrap_or("mlp");

    let mut dataset: Option<String> = None;
    let mut batch_size: usize = 32;
    let mut num_samples: usize = 0;
    let mut hidden_size: usize = 64;
    let mut epochs: usize = 1000;

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

/// Dispatch to the right loader based on dataset name.
pub fn load_dataset(args: &Args) -> Dataset {
    match args.dataset.as_str() {
        "generated" => load_generated_data(args.num_samples),
        "iris" => load_iris_data(),
        "wine-red" => load_wine_data("data/winequality-red.csv"),
        "wine-white" => load_wine_data("data/winequality-white.csv"),
        "breast-cancer" => load_breast_cancer_data(),
        other => {
            eprintln!("Unknown dataset: {}", other);
            std::process::exit(1);
        }
    }
}
