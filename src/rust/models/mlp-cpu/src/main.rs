use nn_common::{
    parse_args, load_dataset, normalize_features, shuffle_data, Xorshift32,
    sgemm_nn, sgemm_tn, sgemm_nt, init_thread_pool,
    bias_relu, bias_softmax, cross_entropy_grad, col_sum, sgd_update,
    relu_backward, xavier_init, OMP_ELEM_THRESHOLD,
};
use rayon::prelude::*;
use std::time::Instant;

const DEFAULT_LR: f32 = 0.02;

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
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;
        let mut rng = Xorshift32::new(seed);

        let mut input_weights = vec![0.0f32; input_size * hidden_size];
        xavier_init(&mut input_weights, input_size, &mut rng);

        let mut output_weights = vec![0.0f32; hidden_size * output_size];
        xavier_init(&mut output_weights, hidden_size, &mut rng);

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

        // hidden[bs, hid] = X[bs, inp] @ W1[inp, hid]
        sgemm_nn(bs, hid, inp, x, &self.input_weights, hidden);

        // Add hidden bias + ReLU
        bias_relu(&mut hidden[..bs * hid], &self.hidden_biases, bs, hid);

        // output[bs, out] = hidden[bs, hid] @ W2[hid, out]
        sgemm_nn(bs, out, hid, &hidden[..bs * hid], &self.output_weights, output);

        // Add output bias + softmax per row
        bias_softmax(&mut output[..bs * out], &self.output_biases, bs, out);
    }

    /// Mini-batch SGD training with cross-entropy loss.
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
        if optimizer == "adam" {
            eprintln!("Warning: Adam optimizer not yet implemented, using SGD");
        }
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

                let x = &inputs[start * inp..];
                let y = &targets[start..];

                // ---- Forward ----
                self.forward_batch(x, bs, &mut hidden, &mut output);

                // ---- Loss + d_output = softmax - one_hot ----
                let batch_loss = cross_entropy_grad(
                    &output[..bs * out], &y[..bs],
                    &mut d_output[..bs * out], bs, out,
                );
                epoch_loss += batch_loss;

                // ---- Backward ----

                // grad_W2[hid, out] = hidden^T @ d_output
                sgemm_tn(hid, out, bs, &hidden, &d_output, &mut grad_w2);

                // grad_b2 = column sum of d_output
                col_sum(&d_output[..bs * out], &mut grad_b2, bs, out);

                // d_hidden = d_output @ W2^T, then ReLU mask
                sgemm_nt(bs, hid, out, &d_output, &self.output_weights, &mut d_hidden);

                let elem = bs as i64 * hid as i64;
                if elem > OMP_ELEM_THRESHOLD {
                    d_hidden[..bs * hid]
                        .par_iter_mut()
                        .zip(hidden[..bs * hid].par_iter())
                        .for_each(|(dh, &h)| {
                            if h <= 0.0 { *dh = 0.0; }
                        });
                } else {
                    relu_backward(&mut d_hidden[..bs * hid], &hidden[..bs * hid]);
                }

                // grad_W1[inp, hid] = X^T @ d_hidden
                sgemm_tn(inp, hid, bs, x, &d_hidden, &mut grad_w1);

                // grad_b1 = column sum of d_hidden
                col_sum(&d_hidden[..bs * hid], &mut grad_b1, bs, hid);

                // ---- SGD update ----
                let lr_s = lr / bs as f32;

                sgd_update(&mut self.input_weights, &grad_w1, lr_s);
                sgd_update(&mut self.output_weights, &grad_w2, lr_s);
                sgd_update(&mut self.hidden_biases, &grad_b1, lr_s);
                sgd_update(&mut self.output_biases, &grad_b2, lr_s);
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

    let mut mlp = Mlp::new(dataset.input_size, args.hidden_size, dataset.output_size);
    let learning_rate = if args.learning_rate > 0.0 { args.learning_rate } else { DEFAULT_LR };

    println!(
        "\nTraining ({} epochs, batch_size={}, hidden={}, lr={:.4})...",
        args.epochs, args.batch_size, args.hidden_size, learning_rate
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
