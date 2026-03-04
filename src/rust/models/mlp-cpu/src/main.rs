use nn_common::{
    parse_args, load_dataset, normalize_features, shuffle_data, Xorshift32,
    sgemm_nn, sgemm_tn, sgemm_nt, init_thread_pool,
    bias_relu, bias_softmax, cross_entropy_grad, col_sum, sgd_update, adam_update,
    relu_backward, xavier_init, OMP_ELEM_THRESHOLD, AdamState,
};
use rayon::prelude::*;
use std::time::Instant;

const DEFAULT_LR: f32 = 0.02;

// ---------------------------------------------------------------------------
//  MLP struct and methods
// ---------------------------------------------------------------------------

struct Mlp {
    layer_sizes: Vec<usize>,    // [input, hid, ..., hid, output]
    weights: Vec<Vec<f32>>,     // weights[i]: [layer_sizes[i] x layer_sizes[i+1]]
    biases: Vec<Vec<f32>>,      // biases[i]:  [layer_sizes[i+1]]
}

impl Mlp {
    /// Allocate parameters and apply Xavier uniform initialization.
    fn new(input_size: usize, hidden_size: usize, output_size: usize,
           num_hidden_layers: usize) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;
        let mut rng = Xorshift32::new(seed);

        let mut layer_sizes = Vec::with_capacity(num_hidden_layers + 2);
        layer_sizes.push(input_size);
        for _ in 0..num_hidden_layers {
            layer_sizes.push(hidden_size);
        }
        layer_sizes.push(output_size);

        let num_layers = layer_sizes.len() - 1;
        let mut weights = Vec::with_capacity(num_layers);
        let mut biases = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let fan_in = layer_sizes[i];
            let fan_out = layer_sizes[i + 1];
            let mut w = vec![0.0f32; fan_in * fan_out];
            xavier_init(&mut w, fan_in, &mut rng);
            weights.push(w);
            biases.push(vec![0.0f32; fan_out]);
        }

        Mlp { layer_sizes, weights, biases }
    }

    fn num_layers(&self) -> usize {
        self.layer_sizes.len() - 1
    }

    fn num_hidden(&self) -> usize {
        self.num_layers() - 1
    }

    /// Batched forward pass storing all hidden activations for backward.
    fn forward_batch(&self, x: &[f32], bs: usize, acts: &mut [Vec<f32>], output: &mut [f32]) {
        let nl = self.num_layers();
        let num_hidden = self.num_hidden();

        for i in 0..nl {
            let in_dim = self.layer_sizes[i];
            let out_dim = self.layer_sizes[i + 1];

            if i < num_hidden {
                // Use split_at_mut to get disjoint borrows of acts[i-1] and acts[i]
                if i == 0 {
                    sgemm_nn(bs, out_dim, in_dim, x, &self.weights[i], &mut acts[i]);
                    bias_relu(&mut acts[i][..bs * out_dim], &self.biases[i], bs, out_dim);
                } else {
                    let (prev, cur_and_rest) = acts.split_at_mut(i);
                    let input_act = &prev[i - 1];
                    let act = &mut cur_and_rest[0];
                    sgemm_nn(bs, out_dim, in_dim, input_act, &self.weights[i], act);
                    bias_relu(&mut act[..bs * out_dim], &self.biases[i], bs, out_dim);
                }
            } else {
                let input_act: &[f32] = if i == 0 { x } else { &acts[i - 1] };
                sgemm_nn(bs, out_dim, in_dim, input_act, &self.weights[i], output);
                bias_softmax(&mut output[..bs * out_dim], &self.biases[i], bs, out_dim);
            }
        }
    }

    /// Forward pass for evaluation (ping-pong buffers, no activation storage).
    fn forward_eval(&self, x: &[f32], bs: usize, output: &mut [f32]) {
        let nl = self.num_layers();
        let num_hidden = self.num_hidden();
        let hid = if num_hidden > 0 { self.layer_sizes[1] } else { 0 };

        // Combined buffer: [buf0 | buf1], split with split_at_mut for disjoint access.
        let buf_len = bs * hid;
        let mut bufs = vec![0.0f32; 2 * buf_len];

        for i in 0..nl {
            let in_dim = self.layer_sizes[i];
            let out_dim = self.layer_sizes[i + 1];

            if i < num_hidden {
                let (b0, b1) = bufs.split_at_mut(buf_len);
                if i == 0 {
                    // input from x, output to buf0 (even layer 0)
                    sgemm_nn(bs, out_dim, in_dim, x, &self.weights[i], b0);
                    bias_relu(&mut b0[..bs * out_dim], &self.biases[i], bs, out_dim);
                } else {
                    // Ping-pong: read from previous, write to current
                    let (src, dst) = if i % 2 == 0 { (b1 as &[f32], b0) } else { (b0 as &[f32], b1) };
                    sgemm_nn(bs, out_dim, in_dim, src, &self.weights[i], dst);
                    bias_relu(&mut dst[..bs * out_dim], &self.biases[i], bs, out_dim);
                }
            } else {
                let input_act: &[f32] = if i == 0 {
                    x
                } else if (i - 1) % 2 == 0 {
                    &bufs[..buf_len]
                } else {
                    &bufs[buf_len..]
                };
                sgemm_nn(bs, out_dim, in_dim, input_act, &self.weights[i], output);
                bias_softmax(&mut output[..bs * out_dim], &self.biases[i], bs, out_dim);
            }
        }
    }

    /// Mini-batch training with cross-entropy loss.
    fn train(
        &mut self,
        inputs: &[f32],
        targets: &[i32],
        num_samples: usize,
        batch_size: usize,
        num_epochs: usize,
        learning_rate: f32,
        optimizer: &str,
        scheduler: &str,
    ) {
        let nl = self.num_layers();
        let num_hidden = self.num_hidden();
        let out = *self.layer_sizes.last().unwrap();
        let hid = if num_hidden > 0 { self.layer_sizes[1] } else { 0 };

        // Hidden activation buffers
        let mut acts: Vec<Vec<f32>> = (0..num_hidden)
            .map(|_| vec![0.0f32; batch_size * hid])
            .collect();

        let mut output = vec![0.0f32; batch_size * out];
        let mut d_output = vec![0.0f32; batch_size * out];

        // Gradient propagation (ping-pong via combined buffer + split_at_mut)
        let max_dim = if hid > out { hid } else { out };
        let grad_buf_len = batch_size * max_dim;
        let mut d_grad_bufs = vec![0.0f32; 2 * grad_buf_len];

        // Per-layer gradients
        let mut grad_w: Vec<Vec<f32>> = (0..nl)
            .map(|i| vec![0.0f32; self.layer_sizes[i] * self.layer_sizes[i + 1]])
            .collect();
        let mut grad_b: Vec<Vec<f32>> = (0..nl)
            .map(|i| vec![0.0f32; self.layer_sizes[i + 1]])
            .collect();

        // Adam state
        let use_adam = optimizer == "adam";
        let mut adam_w: Vec<Option<AdamState>> = (0..nl).map(|i| {
            if use_adam {
                Some(AdamState::new(self.layer_sizes[i] * self.layer_sizes[i + 1]))
            } else {
                None
            }
        }).collect();
        let mut adam_b: Vec<Option<AdamState>> = (0..nl).map(|i| {
            if use_adam {
                Some(AdamState::new(self.layer_sizes[i + 1]))
            } else {
                None
            }
        }).collect();
        let mut step: u32 = 0;

        let num_batches = (num_samples + batch_size - 1) / batch_size;
        let warmup = if scheduler == "cosine" {
            (num_epochs as f32 * 0.05).max(1.0) as usize
        } else {
            0
        };

        for epoch in 0..num_epochs {
            let lr = if scheduler == "cosine" {
                nn_common::cosine_lr(epoch, num_epochs, learning_rate, warmup, 1e-6)
            } else {
                learning_rate
            };
            let mut epoch_loss = 0.0f32;

            for batch in 0..num_batches {
                let start = batch * batch_size;
                let bs = batch_size.min(num_samples - start);
                let inp = self.layer_sizes[0];

                let x = &inputs[start * inp..];
                let y = &targets[start..];

                // ---- Forward ----
                self.forward_batch(x, bs, &mut acts, &mut output);

                // ---- Loss + d_output ----
                let batch_loss = cross_entropy_grad(
                    &output[..bs * out], &y[..bs],
                    &mut d_output[..bs * out], bs, out,
                );
                epoch_loss += batch_loss;

                // ---- Backward ----
                // dcur_id tracks which buffer holds the current gradient
                // 0 = d_output, 1 = d_grad_bufs[..grad_buf_len], 2 = d_grad_bufs[grad_buf_len..]
                let mut dcur_id: u8 = 0;

                for layer in (0..nl).rev() {
                    let fan_in = self.layer_sizes[layer];
                    let fan_out = self.layer_sizes[layer + 1];
                    let input_act: &[f32] = if layer == 0 { x } else { &acts[layer - 1] };

                    let dcur_slice: &[f32] = match dcur_id {
                        0 => &d_output,
                        1 => &d_grad_bufs[..grad_buf_len],
                        _ => &d_grad_bufs[grad_buf_len..],
                    };

                    // grad_W = input_act^T @ dcur
                    sgemm_tn(fan_in, fan_out, bs, input_act, dcur_slice, &mut grad_w[layer]);

                    // grad_b = col_sum(dcur)
                    col_sum(&dcur_slice[..bs * fan_out], &mut grad_b[layer], bs, fan_out);

                    if layer > 0 {
                        // d_next = dcur @ W^T
                        let next_id: u8 = match dcur_id {
                            0 => 1,
                            1 => 2,
                            _ => 1,
                        };

                        {
                            // Use split_at_mut for disjoint access to the two gradient buffers.
                            // When dcur_id == 0, source is d_output (separate), dest is buf 1.
                            let (gb0, gb1) = d_grad_bufs.split_at_mut(grad_buf_len);
                            let (dcur_ref, dnext): (&[f32], &mut [f32]) = match (dcur_id, next_id) {
                                (0, 1) => (&d_output, gb0),
                                (0, _) => (&d_output, gb1),
                                (1, 2) => (gb0 as &[f32], gb1),
                                (2, 1) => (gb1 as &[f32], gb0),
                                // Shouldn't happen, but be safe
                                _ => (&d_output, gb0),
                            };
                            sgemm_nt(bs, fan_in, fan_out, dcur_ref, &self.weights[layer], dnext);

                            let elem = bs as i64 * fan_in as i64;
                            if elem > OMP_ELEM_THRESHOLD {
                                dnext[..bs * fan_in]
                                    .par_iter_mut()
                                    .zip(acts[layer - 1][..bs * fan_in].par_iter())
                                    .for_each(|(dh, &h)| {
                                        if h <= 0.0 { *dh = 0.0; }
                                    });
                            } else {
                                relu_backward(&mut dnext[..bs * fan_in], &acts[layer - 1][..bs * fan_in]);
                            }
                        }
                        dcur_id = next_id;
                    }
                }

                // ---- Parameter update ----
                let lr_s = lr / bs as f32;

                if use_adam {
                    step += 1;
                    for i in 0..nl {
                        adam_update(&mut self.weights[i], &grad_w[i],
                                    adam_w[i].as_mut().unwrap(), lr, 0.9, 0.999, 1e-8, step);
                        adam_update(&mut self.biases[i], &grad_b[i],
                                    adam_b[i].as_mut().unwrap(), lr, 0.9, 0.999, 1e-8, step);
                    }
                } else {
                    for i in 0..nl {
                        sgd_update(&mut self.weights[i], &grad_w[i], lr_s);
                        sgd_update(&mut self.biases[i], &grad_b[i], lr_s);
                    }
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
        let out = *self.layer_sizes.last().unwrap();

        let mut output = vec![0.0f32; num_samples * out];
        self.forward_eval(inputs, num_samples, &mut output);

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
    init_thread_pool();

    let args = parse_args();
    let mut dataset = load_dataset(&args);

    println!(
        "Dataset: {}  ({} samples, {} features, {} classes)",
        args.dataset, dataset.num_samples, dataset.input_size, dataset.output_size
    );

    // Skip z-score normalization for MNIST (pixels already [0,1])
    if args.dataset != "mnist" {
        normalize_features(&mut dataset.inputs, dataset.num_samples, dataset.input_size);
    }
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

    let mut mlp = Mlp::new(dataset.input_size, args.hidden_size, dataset.output_size,
                           args.num_hidden_layers);
    let learning_rate = if args.learning_rate > 0.0 { args.learning_rate } else { DEFAULT_LR };

    println!(
        "\nTraining ({} epochs, batch_size={}, hidden={}, layers={}, lr={:.4})...",
        args.epochs, args.batch_size, args.hidden_size, args.num_hidden_layers, learning_rate
    );

    let t_start = Instant::now();
    mlp.train(
        &dataset.inputs[..train_size * dataset.input_size],
        &dataset.labels[..train_size],
        train_size,
        args.batch_size,
        args.epochs,
        learning_rate,
        &args.optimizer,
        &args.scheduler,
    );
    let t_train = t_start.elapsed().as_secs_f64();

    let t_eval_start = Instant::now();
    let (loss, accuracy) = mlp.evaluate(
        &dataset.inputs[train_size * dataset.input_size..],
        &dataset.labels[train_size..],
        test_size,
    );
    let t_eval = t_eval_start.elapsed().as_secs_f64();

    // Standardized output format -- parsed by benchmark.py
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
