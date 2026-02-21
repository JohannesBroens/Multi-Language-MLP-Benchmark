use mlp_common::{parse_args, load_dataset, normalize_features, shuffle_data, Xorshift32};
use rayon::prelude::*;
use std::time::Instant;

// ---------------------------------------------------------------------------
//  Constants — match C mlp_cpu.c exactly
// ---------------------------------------------------------------------------

/// Tile size for L1 cache residency (~32KB): 3 tiles of 64x64 floats = 48KB.
const TILE: usize = 64;

/// Minimum FLOPs (M*N*K) to justify rayon parallel overhead.
const OMP_FLOP_THRESHOLD: i64 = 100_000;

/// Minimum elements for elementwise parallel (bias+relu, softmax, SGD).
const OMP_ELEM_THRESHOLD: i64 = 200_000;

const LEARNING_RATE: f32 = 0.01;

// ---------------------------------------------------------------------------
//  GEMM — cache-tiled, i-k-j order (matches C sgemm_nn/tn/nt)
// ---------------------------------------------------------------------------

/// C[M,N] = A[M,K] @ B[K,N]  -- cache-friendly i-k-j with optional tiling.
fn sgemm_nn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let flops = m as i64 * n as i64 * k as i64;
    // Slice c to exactly m*n to avoid processing extra rows from oversized buffers
    let c = &mut c[..m * n];

    if k <= TILE && n <= TILE {
        // Small matrices: simple loop, compiler auto-vectorizes j-loop.
        if flops > OMP_FLOP_THRESHOLD {
            c.par_chunks_mut(n).enumerate().for_each(|(i, ci)| {
                for j in 0..n {
                    ci[j] = 0.0;
                }
                for kk in 0..k {
                    let a_ik = a[i * k + kk];
                    let bk = &b[kk * n..kk * n + n];
                    for j in 0..n {
                        ci[j] += a_ik * bk[j];
                    }
                }
            });
        } else {
            for i in 0..m {
                let ci = &mut c[i * n..(i + 1) * n];
                let ai = &a[i * k..];
                for j in 0..n {
                    ci[j] = 0.0;
                }
                for kk in 0..k {
                    let a_ik = ai[kk];
                    let bk = &b[kk * n..kk * n + n];
                    for j in 0..n {
                        ci[j] += a_ik * bk[j];
                    }
                }
            }
        }
    } else {
        // Large matrices: tiled for L1 cache residency.
        for val in c[..m * n].iter_mut() {
            *val = 0.0;
        }

        if flops > OMP_FLOP_THRESHOLD {
            // Parallel over i-tiles: each thread handles a contiguous block of rows.
            let num_i_tiles = (m + TILE - 1) / TILE;
            let tiles: Vec<usize> = (0..num_i_tiles).collect();
            tiles.into_par_iter().for_each(|it| {
                let i0 = it * TILE;
                let imax = (i0 + TILE).min(m);
                // SAFETY: Each thread writes exclusively to rows [i0..imax) of C.
                // No two threads share the same row range since i-tiles are disjoint.
                let c_ptr = c.as_ptr() as *mut f32;
                for k0 in (0..k).step_by(TILE) {
                    let kmax = (k0 + TILE).min(k);
                    for j0 in (0..n).step_by(TILE) {
                        let jmax = (j0 + TILE).min(n);
                        for i in i0..imax {
                            for kk in k0..kmax {
                                let a_ik = a[i * k + kk];
                                for j in j0..jmax {
                                    // SAFETY: i is in [i0..imax), exclusive per thread.
                                    // Index i*n+j is within bounds [0, m*n).
                                    unsafe {
                                        *c_ptr.add(i * n + j) += a_ik * b[kk * n + j];
                                    }
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

/// C[M,N] = A^T[M,K] @ B[K,N],  A stored as [K,M].
fn sgemm_tn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let flops = m as i64 * n as i64 * k as i64;
    let c = &mut c[..m * n];

    if flops > OMP_FLOP_THRESHOLD {
        c.par_chunks_mut(n).enumerate().for_each(|(i, ci)| {
            for j in 0..n {
                ci[j] = 0.0;
            }
            for kk in 0..k {
                let a_ki = a[kk * m + i];
                let bk = &b[kk * n..kk * n + n];
                for j in 0..n {
                    ci[j] += a_ki * bk[j];
                }
            }
        });
    } else {
        for i in 0..m {
            let ci = &mut c[i * n..(i + 1) * n];
            for j in 0..n {
                ci[j] = 0.0;
            }
            for kk in 0..k {
                let a_ki = a[kk * m + i];
                let bk = &b[kk * n..kk * n + n];
                for j in 0..n {
                    ci[j] += a_ki * bk[j];
                }
            }
        }
    }
}

/// C[M,N] = A[M,K] @ B^T[K,N],  B stored as [N,K].
fn sgemm_nt(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let flops = m as i64 * n as i64 * k as i64;
    let c = &mut c[..m * n];

    if flops > OMP_FLOP_THRESHOLD {
        c.par_chunks_mut(n).enumerate().for_each(|(i, ci)| {
            let ai = &a[i * k..i * k + k];
            for j in 0..n {
                let mut sum = 0.0f32;
                let bj = &b[j * k..j * k + k];
                for kk in 0..k {
                    sum += ai[kk] * bj[kk];
                }
                ci[j] = sum;
            }
        });
    } else {
        for i in 0..m {
            let ai = &a[i * k..i * k + k];
            for j in 0..n {
                let mut sum = 0.0f32;
                let bj = &b[j * k..j * k + k];
                for kk in 0..k {
                    sum += ai[kk] * bj[kk];
                }
                c[i * n + j] = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  MLP struct and methods
// ---------------------------------------------------------------------------

struct Mlp {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    input_weights: Vec<f32>,   // [input_size  x hidden_size]
    output_weights: Vec<f32>,  // [hidden_size x output_size]
    hidden_biases: Vec<f32>,   // [hidden_size]
    output_biases: Vec<f32>,   // [output_size]
}

impl Mlp {
    /// Allocate parameters and apply Xavier uniform initialization.
    /// Uses the same xorshift32 PRNG and rand_float as the C version.
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;
        let mut rng = Xorshift32::new(seed);

        // Xavier uniform: Uniform(-sqrt(2/fan_in), sqrt(2/fan_in))
        // C uses: (rand_float(&rng) * 2.0 - 1.0) * scale
        // where rand_float = (xorshift32(state) & 0x7FFFFFFF) / 0x7FFFFFFF
        let s1 = (2.0f32 / input_size as f32).sqrt();
        let mut input_weights = vec![0.0f32; input_size * hidden_size];
        for w in input_weights.iter_mut() {
            let r = (rng.next_u32() & 0x7FFF_FFFF) as f32 / 0x7FFF_FFFF_u32 as f32;
            *w = (r * 2.0 - 1.0) * s1;
        }

        let s2 = (2.0f32 / hidden_size as f32).sqrt();
        let mut output_weights = vec![0.0f32; hidden_size * output_size];
        for w in output_weights.iter_mut() {
            let r = (rng.next_u32() & 0x7FFF_FFFF) as f32 / 0x7FFF_FFFF_u32 as f32;
            *w = (r * 2.0 - 1.0) * s2;
        }

        Mlp {
            input_size,
            hidden_size,
            output_size,
            input_weights,
            output_weights,
            hidden_biases: vec![0.0; hidden_size],
            output_biases: vec![0.0; output_size],
        }
    }

    /// Batched forward pass: hidden = relu(X @ W1 + b1), output = softmax(hidden @ W2 + b2).
    fn forward_batch(&self, x: &[f32], bs: usize, hidden: &mut [f32], output: &mut [f32]) {
        let inp = self.input_size;
        let hid = self.hidden_size;
        let out = self.output_size;

        // Slice buffers to actual batch size to avoid processing extra rows
        let hidden = &mut hidden[..bs * hid];
        let output = &mut output[..bs * out];

        // hidden[bs, hid] = X[bs, inp] @ W1[inp, hid]
        sgemm_nn(bs, hid, inp, x, &self.input_weights, hidden);

        // Add hidden bias + ReLU
        let elem = bs as i64 * hid as i64;
        if elem > OMP_ELEM_THRESHOLD {
            hidden.par_chunks_mut(hid).for_each(|h| {
                for j in 0..hid {
                    h[j] += self.hidden_biases[j];
                    if h[j] < 0.0 {
                        h[j] = 0.0;
                    }
                }
            });
        } else {
            for i in 0..bs {
                let h = &mut hidden[i * hid..(i + 1) * hid];
                for j in 0..hid {
                    h[j] += self.hidden_biases[j];
                    if h[j] < 0.0 {
                        h[j] = 0.0;
                    }
                }
            }
        }

        // output[bs, out] = hidden[bs, hid] @ W2[hid, out]
        sgemm_nn(bs, out, hid, hidden, &self.output_weights, output);

        // Add output bias + numerically stable softmax per row
        let elem = bs as i64 * out as i64;
        if elem > OMP_ELEM_THRESHOLD {
            output.par_chunks_mut(out).for_each(|o| {
                let mut max_val = -1e30f32;
                for j in 0..out {
                    o[j] += self.output_biases[j];
                    if o[j] > max_val {
                        max_val = o[j];
                    }
                }
                let mut sum = 0.0f32;
                for j in 0..out {
                    o[j] = (o[j] - max_val).exp();
                    sum += o[j];
                }
                for j in 0..out {
                    o[j] /= sum;
                }
            });
        } else {
            for i in 0..bs {
                let o = &mut output[i * out..(i + 1) * out];
                let mut max_val = -1e30f32;
                for j in 0..out {
                    o[j] += self.output_biases[j];
                    if o[j] > max_val {
                        max_val = o[j];
                    }
                }
                let mut sum = 0.0f32;
                for j in 0..out {
                    o[j] = (o[j] - max_val).exp();
                    sum += o[j];
                }
                for j in 0..out {
                    o[j] /= sum;
                }
            }
        }
    }

    /// Mini-batch SGD training with cross-entropy loss.
    fn train(
        &mut self,
        inputs: &[f32],
        targets: &[i32],
        num_samples: usize,
        batch_size: usize,
        num_epochs: usize,
    ) {
        let inp = self.input_size;
        let hid = self.hidden_size;
        let out = self.output_size;

        let mut hidden = vec![0.0f32; batch_size * hid];
        let mut output = vec![0.0f32; batch_size * out];
        let mut d_output = vec![0.0f32; batch_size * out];
        let mut d_hidden = vec![0.0f32; batch_size * hid];
        let mut grad_w1 = vec![0.0f32; inp * hid];
        let mut grad_w2 = vec![0.0f32; hid * out];
        let mut grad_b1 = vec![0.0f32; hid];
        let mut grad_b2 = vec![0.0f32; out];

        let num_batches = (num_samples + batch_size - 1) / batch_size;

        for epoch in 0..num_epochs {
            let mut epoch_loss = 0.0f32;

            for batch in 0..num_batches {
                let start = batch * batch_size;
                let bs = batch_size.min(num_samples - start);

                let x = &inputs[start * inp..];
                let y = &targets[start..];

                // ---- Forward ----
                self.forward_batch(x, bs, &mut hidden, &mut output);

                // ---- Loss + d_output = softmax - one_hot ----
                let mut batch_loss = 0.0f32;
                for i in 0..bs {
                    let o = &output[i * out..(i + 1) * out];
                    let d = &mut d_output[i * out..(i + 1) * out];
                    batch_loss -= (o[y[i] as usize] + 1e-7).ln();
                    for j in 0..out {
                        d[j] = o[j] - if y[i] == j as i32 { 1.0 } else { 0.0 };
                    }
                }
                epoch_loss += batch_loss;

                // ---- Backward ----

                // grad_W2[hid, out] = hidden^T @ d_output
                sgemm_tn(hid, out, bs, &hidden, &d_output, &mut grad_w2);

                // grad_b2 = column sum of d_output
                for j in 0..out {
                    grad_b2[j] = 0.0;
                }
                for i in 0..bs {
                    let d = &d_output[i * out..(i + 1) * out];
                    for j in 0..out {
                        grad_b2[j] += d[j];
                    }
                }

                // d_hidden = d_output @ W2^T, then ReLU mask
                sgemm_nt(bs, hid, out, &d_output, &self.output_weights, &mut d_hidden);

                let elem = bs as i64 * hid as i64;
                if elem > OMP_ELEM_THRESHOLD {
                    d_hidden[..bs * hid]
                        .par_iter_mut()
                        .zip(hidden[..bs * hid].par_iter())
                        .for_each(|(dh, &h)| {
                            if h <= 0.0 {
                                *dh = 0.0;
                            }
                        });
                } else {
                    for i in 0..bs * hid {
                        if hidden[i] <= 0.0 {
                            d_hidden[i] = 0.0;
                        }
                    }
                }

                // grad_W1[inp, hid] = X^T @ d_hidden
                sgemm_tn(inp, hid, bs, x, &d_hidden, &mut grad_w1);

                // grad_b1 = column sum of d_hidden
                for j in 0..hid {
                    grad_b1[j] = 0.0;
                }
                for i in 0..bs {
                    let d = &d_hidden[i * hid..(i + 1) * hid];
                    for j in 0..hid {
                        grad_b1[j] += d[j];
                    }
                }

                // ---- SGD update ----
                let lr_s = LEARNING_RATE / bs as f32;

                let wt_elem = inp as i64 * hid as i64;
                if wt_elem > OMP_ELEM_THRESHOLD {
                    self.input_weights
                        .par_iter_mut()
                        .zip(grad_w1.par_iter())
                        .for_each(|(w, &g)| {
                            *w -= lr_s * g;
                        });
                } else {
                    for i in 0..inp * hid {
                        self.input_weights[i] -= lr_s * grad_w1[i];
                    }
                }

                let wt_elem = hid as i64 * out as i64;
                if wt_elem > OMP_ELEM_THRESHOLD {
                    self.output_weights
                        .par_iter_mut()
                        .zip(grad_w2.par_iter())
                        .for_each(|(w, &g)| {
                            *w -= lr_s * g;
                        });
                } else {
                    for i in 0..hid * out {
                        self.output_weights[i] -= lr_s * grad_w2[i];
                    }
                }

                for i in 0..hid {
                    self.hidden_biases[i] -= lr_s * grad_b1[i];
                }
                for i in 0..out {
                    self.output_biases[i] -= lr_s * grad_b2[i];
                }
            }

            if epoch % 100 == 0 {
                println!(
                    "  Epoch {:4}  loss: {:.4}",
                    epoch,
                    epoch_loss / num_samples as f32
                );
            }
        }
    }

    /// Evaluate on test set: average cross-entropy loss and accuracy %.
    fn evaluate(&self, inputs: &[f32], targets: &[i32], num_samples: usize) -> (f32, f32) {
        let hid = self.hidden_size;
        let out = self.output_size;

        let mut hidden = vec![0.0f32; num_samples * hid];
        let mut output = vec![0.0f32; num_samples * out];

        self.forward_batch(inputs, num_samples, &mut hidden, &mut output);

        let mut total_loss = 0.0f32;
        let mut correct = 0i32;

        for i in 0..num_samples {
            let o = &output[i * out..(i + 1) * out];
            total_loss -= (o[targets[i] as usize] + 1e-7).ln();

            let mut pred = 0;
            let mut mx = o[0];
            for j in 1..out {
                if o[j] > mx {
                    mx = o[j];
                    pred = j;
                }
            }
            if pred as i32 == targets[i] {
                correct += 1;
            }
        }

        let loss = total_loss / num_samples as f32;
        let accuracy = correct as f32 / num_samples as f32 * 100.0;
        (loss, accuracy)
    }
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------

fn main() {
    // Set rayon thread pool to physical cores only (avoid HT contention on FPU).
    let physical_cores = num_cpus::get_physical();
    rayon::ThreadPoolBuilder::new()
        .num_threads(physical_cores)
        .build_global()
        .unwrap();

    let args = parse_args();
    let mut dataset = load_dataset(&args);

    println!(
        "Dataset: {}  ({} samples, {} features, {} classes)",
        args.dataset, dataset.num_samples, dataset.input_size, dataset.output_size
    );

    normalize_features(&mut dataset.inputs, dataset.num_samples, dataset.input_size);
    shuffle_data(
        &mut dataset.inputs,
        &mut dataset.labels,
        dataset.num_samples,
        dataset.input_size,
    );

    // 80/20 train/test split
    let train_size = (dataset.num_samples as f32 * 0.8) as usize;
    let test_size = dataset.num_samples - train_size;

    println!("Train: {} samples, Test: {} samples", train_size, test_size);

    let mut mlp = Mlp::new(dataset.input_size, args.hidden_size, dataset.output_size);

    println!(
        "\nTraining ({} epochs, batch_size={}, hidden={}, lr={:.4})...",
        args.epochs, args.batch_size, args.hidden_size, LEARNING_RATE
    );

    let t_start = Instant::now();
    mlp.train(
        &dataset.inputs[..train_size * dataset.input_size],
        &dataset.labels[..train_size],
        train_size,
        args.batch_size,
        args.epochs,
    );
    let t_train = t_start.elapsed().as_secs_f64();

    let t_eval_start = Instant::now();
    let (loss, accuracy) = mlp.evaluate(
        &dataset.inputs[train_size * dataset.input_size..],
        &dataset.labels[train_size..],
        test_size,
    );
    let t_eval = t_eval_start.elapsed().as_secs_f64();

    // Standardized output format — parsed by benchmark.py
    println!("\n=== Results on {} ===", args.dataset);
    println!("Test Loss:     {:.4}", loss);
    println!("Test Accuracy: {:.2}%", accuracy);
    println!("Train time:    {:.3} s", t_train);
    println!("Eval time:     {:.3} s", t_eval);
    println!(
        "Throughput:    {:.0} samples/s",
        train_size as f64 * args.epochs as f64 / t_train
    );
}
