use nn_common::{
    parse_args, load_dataset, shuffle_data, Xorshift32,
    sgemm_nn, sgemm_tn, sgemm_nt, init_thread_pool,
    bias_relu, bias_softmax, cross_entropy_grad, col_sum, sgd_update,
    relu_forward, relu_backward, xavier_init,
};
use std::time::Instant;

const LEARNING_RATE: f32 = 0.01;

// ---------------------------------------------------------------------------
//  LeNet-5 layer dimensions
// ---------------------------------------------------------------------------

const IN_C: usize = 1;
const IN_H: usize = 28;
const IN_W: usize = 28;

const C1_OUT: usize = 6;
const C1_IN: usize = 1;
const C1_K: usize = 5;
const C1_OH: usize = IN_H - C1_K + 1;   // 24
const C1_OW: usize = IN_W - C1_K + 1;   // 24
const C1_COL_ROWS: usize = C1_IN * C1_K * C1_K;  // 25
const C1_COL_COLS: usize = C1_OH * C1_OW;         // 576

const P1_SIZE: usize = 2;
const P1_H: usize = C1_OH / P1_SIZE;   // 12
const P1_W: usize = C1_OW / P1_SIZE;   // 12

const C2_OUT: usize = 16;
const C2_IN: usize = 6;
const C2_K: usize = 5;
const C2_OH: usize = P1_H - C2_K + 1;   // 8
const C2_OW: usize = P1_W - C2_K + 1;   // 8
const C2_COL_ROWS: usize = C2_IN * C2_K * C2_K;  // 150
const C2_COL_COLS: usize = C2_OH * C2_OW;         // 64

const P2_SIZE: usize = 2;
const P2_H: usize = C2_OH / P2_SIZE;   // 4
const P2_W: usize = C2_OW / P2_SIZE;   // 4

const FLAT_SIZE: usize = C2_OUT * P2_H * P2_W;  // 256

const FC1_OUT: usize = 120;
const FC2_OUT: usize = 84;
const NUM_CLASSES: usize = 10;

// ---------------------------------------------------------------------------
//  im2col / col2im / avg_pool
// ---------------------------------------------------------------------------

/// im2col: unfold image patches into columns for GEMM-based convolution.
/// Input shape: [C, H, W]. Output shape: [C*kH*kW, OH*OW].
fn im2col(input: &[f32], c: usize, h: usize, w: usize,
          kh: usize, kw: usize, stride: usize, col: &mut [f32]) {
    let oh = (h - kh) / stride + 1;
    let ow = (w - kw) / stride + 1;
    let mut col_row = 0;

    for ci in 0..c {
        for khi in 0..kh {
            for kwi in 0..kw {
                for ohi in 0..oh {
                    for owi in 0..ow {
                        let hi = ohi * stride + khi;
                        let wi = owi * stride + kwi;
                        col[col_row * (oh * ow) + ohi * ow + owi] =
                            input[ci * h * w + hi * w + wi];
                    }
                }
                col_row += 1;
            }
        }
    }
}

/// col2im: inverse of im2col. Accumulates into output (caller must zero first).
fn col2im(col: &[f32], c: usize, h: usize, w: usize,
          kh: usize, kw: usize, stride: usize, output: &mut [f32]) {
    let oh = (h - kh) / stride + 1;
    let ow = (w - kw) / stride + 1;
    let mut col_row = 0;

    for ci in 0..c {
        for khi in 0..kh {
            for kwi in 0..kw {
                for ohi in 0..oh {
                    for owi in 0..ow {
                        let hi = ohi * stride + khi;
                        let wi = owi * stride + kwi;
                        output[ci * h * w + hi * w + wi] +=
                            col[col_row * (oh * ow) + ohi * ow + owi];
                    }
                }
                col_row += 1;
            }
        }
    }
}

/// Average pooling forward.
fn avg_pool_forward(input: &[f32], output: &mut [f32],
                    c: usize, h: usize, w: usize, pool_size: usize) {
    let oh = h / pool_size;
    let ow = w / pool_size;
    let scale = 1.0 / (pool_size * pool_size) as f32;

    for ci in 0..c {
        for ohi in 0..oh {
            for owi in 0..ow {
                let mut sum = 0.0f32;
                for ph in 0..pool_size {
                    for pw in 0..pool_size {
                        let hi = ohi * pool_size + ph;
                        let wi = owi * pool_size + pw;
                        sum += input[ci * h * w + hi * w + wi];
                    }
                }
                output[ci * oh * ow + ohi * ow + owi] = sum * scale;
            }
        }
    }
}

/// Average pooling backward.
fn avg_pool_backward(grad_output: &[f32], grad_input: &mut [f32],
                     c: usize, h: usize, w: usize, pool_size: usize) {
    let oh = h / pool_size;
    let ow = w / pool_size;
    let scale = 1.0 / (pool_size * pool_size) as f32;

    for ci in 0..c {
        for ohi in 0..oh {
            for owi in 0..ow {
                let g = grad_output[ci * oh * ow + ohi * ow + owi] * scale;
                for ph in 0..pool_size {
                    for pw in 0..pool_size {
                        let hi = ohi * pool_size + ph;
                        let wi = owi * pool_size + pw;
                        grad_input[ci * h * w + hi * w + wi] += g;
                    }
                }
            }
        }
    }
}

/// Add bias to [channels, spatial] matrix.
fn add_bias(x: &mut [f32], bias: &[f32], channels: usize, spatial: usize) {
    for c in 0..channels {
        let b = bias[c];
        for j in 0..spatial {
            x[c * spatial + j] += b;
        }
    }
}

// ---------------------------------------------------------------------------
//  Workspace layout offsets (same as C version)
// ---------------------------------------------------------------------------

const WS_COL1: usize = 0;
const WS_CONV1: usize = C1_COL_ROWS * C1_COL_COLS;
const WS_POOL1: usize = WS_CONV1 + C1_OUT * C1_COL_COLS;
const WS_COL2: usize = WS_POOL1 + C1_OUT * P1_H * P1_W;
const WS_CONV2: usize = WS_COL2 + C2_COL_ROWS * C2_COL_COLS;
const WS_POOL2: usize = WS_CONV2 + C2_OUT * C2_COL_COLS;
const WS_SIZE: usize = WS_POOL2 + FLAT_SIZE;

// ---------------------------------------------------------------------------
//  CNN struct
// ---------------------------------------------------------------------------

struct Cnn {
    conv1_weights: Vec<f32>,  // [6, 25]
    conv1_biases: Vec<f32>,   // [6]
    conv2_weights: Vec<f32>,  // [16, 150]
    conv2_biases: Vec<f32>,   // [16]
    fc1_weights: Vec<f32>,    // [256, 120]
    fc1_biases: Vec<f32>,     // [120]
    fc2_weights: Vec<f32>,    // [120, 84]
    fc2_biases: Vec<f32>,     // [84]
    out_weights: Vec<f32>,    // [84, 10]
    out_biases: Vec<f32>,     // [10]
}

impl Cnn {
    fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;
        let mut rng = Xorshift32::new(seed);

        let mut conv1_weights = vec![0.0f32; C1_OUT * C1_COL_ROWS];
        xavier_init(&mut conv1_weights, C1_COL_ROWS, &mut rng);
        let mut conv2_weights = vec![0.0f32; C2_OUT * C2_COL_ROWS];
        xavier_init(&mut conv2_weights, C2_COL_ROWS, &mut rng);
        let mut fc1_weights = vec![0.0f32; FLAT_SIZE * FC1_OUT];
        xavier_init(&mut fc1_weights, FLAT_SIZE, &mut rng);
        let mut fc2_weights = vec![0.0f32; FC1_OUT * FC2_OUT];
        xavier_init(&mut fc2_weights, FC1_OUT, &mut rng);
        let mut out_weights = vec![0.0f32; FC2_OUT * NUM_CLASSES];
        xavier_init(&mut out_weights, FC2_OUT, &mut rng);

        Cnn {
            conv1_weights,
            conv1_biases: vec![0.0; C1_OUT],
            conv2_weights,
            conv2_biases: vec![0.0; C2_OUT],
            fc1_weights,
            fc1_biases: vec![0.0; FC1_OUT],
            fc2_weights,
            fc2_biases: vec![0.0; FC2_OUT],
            out_weights,
            out_biases: vec![0.0; NUM_CLASSES],
        }
    }

    /// Per-sample conv forward pass.
    /// Uses separate buffers to avoid borrow-checker issues with overlapping slices.
    fn conv_forward_sample(&self, input: &[f32], ws: &mut [f32]) {
        // Use raw pointer for reads from ws while mutating non-overlapping parts.
        // The workspace regions don't overlap (guaranteed by the offset layout).
        let ws_ptr = ws.as_ptr();

        // Conv1: im2col -> GEMM -> bias + ReLU -> pool
        im2col(input, C1_IN, IN_H, IN_W, C1_K, C1_K, 1,
               &mut ws[WS_COL1..WS_COL1 + C1_COL_ROWS * C1_COL_COLS]);

        // col1 (read) and conv1_out (write) don't overlap
        let col1_slice = unsafe {
            std::slice::from_raw_parts(ws_ptr.add(WS_COL1), C1_COL_ROWS * C1_COL_COLS)
        };
        sgemm_nn(C1_OUT, C1_COL_COLS, C1_COL_ROWS,
                 &self.conv1_weights, col1_slice,
                 &mut ws[WS_CONV1..WS_CONV1 + C1_OUT * C1_COL_COLS]);

        add_bias(&mut ws[WS_CONV1..WS_CONV1 + C1_OUT * C1_COL_COLS],
                 &self.conv1_biases, C1_OUT, C1_COL_COLS);
        relu_forward(&mut ws[WS_CONV1..WS_CONV1 + C1_OUT * C1_COL_COLS]);

        // conv1_out (read) and pool1 (write) don't overlap
        let conv1_slice = unsafe {
            std::slice::from_raw_parts(ws_ptr.add(WS_CONV1), C1_OUT * C1_COL_COLS)
        };
        avg_pool_forward(conv1_slice,
                         &mut ws[WS_POOL1..WS_POOL1 + C1_OUT * P1_H * P1_W],
                         C1_OUT, C1_OH, C1_OW, P1_SIZE);

        // Conv2: im2col -> GEMM -> bias + ReLU -> pool
        // pool1 (read) and col2 (write) don't overlap
        let pool1_slice = unsafe {
            std::slice::from_raw_parts(ws_ptr.add(WS_POOL1), C2_IN * P1_H * P1_W)
        };
        im2col(pool1_slice, C2_IN, P1_H, P1_W, C2_K, C2_K, 1,
               &mut ws[WS_COL2..WS_COL2 + C2_COL_ROWS * C2_COL_COLS]);

        let col2_slice = unsafe {
            std::slice::from_raw_parts(ws_ptr.add(WS_COL2), C2_COL_ROWS * C2_COL_COLS)
        };
        sgemm_nn(C2_OUT, C2_COL_COLS, C2_COL_ROWS,
                 &self.conv2_weights, col2_slice,
                 &mut ws[WS_CONV2..WS_CONV2 + C2_OUT * C2_COL_COLS]);

        add_bias(&mut ws[WS_CONV2..WS_CONV2 + C2_OUT * C2_COL_COLS],
                 &self.conv2_biases, C2_OUT, C2_COL_COLS);
        relu_forward(&mut ws[WS_CONV2..WS_CONV2 + C2_OUT * C2_COL_COLS]);

        let conv2_slice = unsafe {
            std::slice::from_raw_parts(ws_ptr.add(WS_CONV2), C2_OUT * C2_COL_COLS)
        };
        avg_pool_forward(conv2_slice,
                         &mut ws[WS_POOL2..WS_POOL2 + FLAT_SIZE],
                         C2_OUT, C2_OH, C2_OW, P2_SIZE);
    }

    /// Batched forward: per-sample conv, then batched FC.
    fn forward_batch(&self, inputs: &[f32], bs: usize,
                     ws_all: &mut [f32], flat: &mut [f32],
                     fc1_out: &mut [f32], fc2_out: &mut [f32], out: &mut [f32]) {
        let input_size = IN_C * IN_H * IN_W;

        for b in 0..bs {
            let img = &inputs[b * input_size..(b + 1) * input_size];
            let ws = &mut ws_all[b * WS_SIZE..(b + 1) * WS_SIZE];
            self.conv_forward_sample(img, ws);
            // Copy pool2 to flat
            flat[b * FLAT_SIZE..(b + 1) * FLAT_SIZE]
                .copy_from_slice(&ws[WS_POOL2..WS_POOL2 + FLAT_SIZE]);
        }

        // FC1
        sgemm_nn(bs, FC1_OUT, FLAT_SIZE, flat, &self.fc1_weights, fc1_out);
        bias_relu(fc1_out, &self.fc1_biases, bs, FC1_OUT);

        // FC2
        sgemm_nn(bs, FC2_OUT, FC1_OUT, fc1_out, &self.fc2_weights, fc2_out);
        bias_relu(fc2_out, &self.fc2_biases, bs, FC2_OUT);

        // Output
        sgemm_nn(bs, NUM_CLASSES, FC2_OUT, fc2_out, &self.out_weights, out);
        bias_softmax(out, &self.out_biases, bs, NUM_CLASSES);
    }

    fn train(&mut self, inputs: &[f32], targets: &[i32], num_samples: usize,
             batch_size: usize, num_epochs: usize, learning_rate: f32) {
        let input_size = IN_C * IN_H * IN_W;

        // Forward workspace
        let mut ws_all = vec![0.0f32; batch_size * WS_SIZE];
        let mut flat = vec![0.0f32; batch_size * FLAT_SIZE];
        let mut fc1_out = vec![0.0f32; batch_size * FC1_OUT];
        let mut fc2_out = vec![0.0f32; batch_size * FC2_OUT];
        let mut out = vec![0.0f32; batch_size * NUM_CLASSES];

        // FC gradients
        let mut d_out = vec![0.0f32; batch_size * NUM_CLASSES];
        let mut d_fc2 = vec![0.0f32; batch_size * FC2_OUT];
        let mut d_fc1 = vec![0.0f32; batch_size * FC1_OUT];
        let mut d_flat = vec![0.0f32; batch_size * FLAT_SIZE];

        let mut grad_out_w = vec![0.0f32; FC2_OUT * NUM_CLASSES];
        let mut grad_out_b = vec![0.0f32; NUM_CLASSES];
        let mut grad_fc2_w = vec![0.0f32; FC1_OUT * FC2_OUT];
        let mut grad_fc2_b = vec![0.0f32; FC2_OUT];
        let mut grad_fc1_w = vec![0.0f32; FLAT_SIZE * FC1_OUT];
        let mut grad_fc1_b = vec![0.0f32; FC1_OUT];

        // Conv gradients
        let mut grad_conv2_w = vec![0.0f32; C2_OUT * C2_COL_ROWS];
        let mut grad_conv2_b = vec![0.0f32; C2_OUT];
        let mut grad_conv1_w = vec![0.0f32; C1_OUT * C1_COL_ROWS];
        let mut grad_conv1_b = vec![0.0f32; C1_OUT];

        // Per-sample backward workspace
        let mut d_pool2 = vec![0.0f32; FLAT_SIZE];
        let mut d_conv2_out = vec![0.0f32; C2_OUT * C2_COL_COLS];
        let mut d_col2 = vec![0.0f32; C2_COL_ROWS * C2_COL_COLS];
        let mut d_pool1 = vec![0.0f32; C2_IN * P1_H * P1_W];
        let mut d_conv1_out = vec![0.0f32; C1_OUT * C1_COL_COLS];

        let mut tmp_grad_w2 = vec![0.0f32; C2_OUT * C2_COL_ROWS];
        let mut tmp_grad_w1 = vec![0.0f32; C1_OUT * C1_COL_ROWS];

        let num_batches = (num_samples + batch_size - 1) / batch_size;

        for epoch in 0..num_epochs {
            let mut epoch_loss = 0.0f32;

            for batch in 0..num_batches {
                let start = batch * batch_size;
                let bs = batch_size.min(num_samples - start);

                let x = &inputs[start * input_size..];
                let y = &targets[start..];

                // Forward
                self.forward_batch(x, bs, &mut ws_all, &mut flat,
                                   &mut fc1_out, &mut fc2_out, &mut out);

                // Loss + output gradient
                let batch_loss = cross_entropy_grad(
                    &out[..bs * NUM_CLASSES], &y[..bs],
                    &mut d_out[..bs * NUM_CLASSES], bs, NUM_CLASSES,
                );
                epoch_loss += batch_loss;

                // FC backward
                sgemm_tn(FC2_OUT, NUM_CLASSES, bs, &fc2_out, &d_out, &mut grad_out_w);
                col_sum(&d_out[..bs * NUM_CLASSES], &mut grad_out_b, bs, NUM_CLASSES);

                sgemm_nt(bs, FC2_OUT, NUM_CLASSES, &d_out, &self.out_weights, &mut d_fc2);
                relu_backward(&mut d_fc2[..bs * FC2_OUT], &fc2_out[..bs * FC2_OUT]);

                sgemm_tn(FC1_OUT, FC2_OUT, bs, &fc1_out, &d_fc2, &mut grad_fc2_w);
                col_sum(&d_fc2[..bs * FC2_OUT], &mut grad_fc2_b, bs, FC2_OUT);

                sgemm_nt(bs, FC1_OUT, FC2_OUT, &d_fc2, &self.fc2_weights, &mut d_fc1);
                relu_backward(&mut d_fc1[..bs * FC1_OUT], &fc1_out[..bs * FC1_OUT]);

                sgemm_tn(FLAT_SIZE, FC1_OUT, bs, &flat, &d_fc1, &mut grad_fc1_w);
                col_sum(&d_fc1[..bs * FC1_OUT], &mut grad_fc1_b, bs, FC1_OUT);

                sgemm_nt(bs, FLAT_SIZE, FC1_OUT, &d_fc1, &self.fc1_weights, &mut d_flat);

                // Conv backward (per-sample)
                for v in grad_conv2_w.iter_mut() { *v = 0.0; }
                for v in grad_conv2_b.iter_mut() { *v = 0.0; }
                for v in grad_conv1_w.iter_mut() { *v = 0.0; }
                for v in grad_conv1_b.iter_mut() { *v = 0.0; }

                for b in 0..bs {
                    let ws = &ws_all[b * WS_SIZE..(b + 1) * WS_SIZE];
                    let col1 = &ws[WS_COL1..WS_COL1 + C1_COL_ROWS * C1_COL_COLS];
                    let conv1_out_buf = &ws[WS_CONV1..WS_CONV1 + C1_OUT * C1_COL_COLS];
                    let col2 = &ws[WS_COL2..WS_COL2 + C2_COL_ROWS * C2_COL_COLS];
                    let conv2_out_buf = &ws[WS_CONV2..WS_CONV2 + C2_OUT * C2_COL_COLS];

                    // d_flat[b] -> d_pool2
                    d_pool2.copy_from_slice(&d_flat[b * FLAT_SIZE..(b + 1) * FLAT_SIZE]);

                    // Pool2 backward
                    for v in d_conv2_out.iter_mut() { *v = 0.0; }
                    avg_pool_backward(&d_pool2, &mut d_conv2_out,
                                      C2_OUT, C2_OH, C2_OW, P2_SIZE);

                    // ReLU backward for conv2
                    relu_backward(&mut d_conv2_out, conv2_out_buf);

                    // Conv2 weight gradient
                    sgemm_nt(C2_OUT, C2_COL_ROWS, C2_COL_COLS,
                             &d_conv2_out, col2, &mut tmp_grad_w2);
                    for (g, &t) in grad_conv2_w.iter_mut().zip(tmp_grad_w2.iter()) {
                        *g += t;
                    }

                    // Conv2 bias gradient
                    for c in 0..C2_OUT {
                        let mut sum = 0.0f32;
                        for j in 0..C2_COL_COLS {
                            sum += d_conv2_out[c * C2_COL_COLS + j];
                        }
                        grad_conv2_b[c] += sum;
                    }

                    // Conv2 input gradient
                    sgemm_tn(C2_COL_ROWS, C2_COL_COLS, C2_OUT,
                             &self.conv2_weights, &d_conv2_out, &mut d_col2);

                    // col2im -> d_pool1
                    for v in d_pool1.iter_mut() { *v = 0.0; }
                    col2im(&d_col2, C2_IN, P1_H, P1_W, C2_K, C2_K, 1, &mut d_pool1);

                    // Pool1 backward
                    for v in d_conv1_out.iter_mut() { *v = 0.0; }
                    avg_pool_backward(&d_pool1, &mut d_conv1_out,
                                      C1_OUT, C1_OH, C1_OW, P1_SIZE);

                    // ReLU backward for conv1
                    relu_backward(&mut d_conv1_out, conv1_out_buf);

                    // Conv1 weight gradient
                    sgemm_nt(C1_OUT, C1_COL_ROWS, C1_COL_COLS,
                             &d_conv1_out, col1, &mut tmp_grad_w1);
                    for (g, &t) in grad_conv1_w.iter_mut().zip(tmp_grad_w1.iter()) {
                        *g += t;
                    }

                    // Conv1 bias gradient
                    for c in 0..C1_OUT {
                        let mut sum = 0.0f32;
                        for j in 0..C1_COL_COLS {
                            sum += d_conv1_out[c * C1_COL_COLS + j];
                        }
                        grad_conv1_b[c] += sum;
                    }
                }

                // SGD update
                let lr_s = learning_rate / bs as f32;

                sgd_update(&mut self.out_weights, &grad_out_w, lr_s);
                sgd_update(&mut self.out_biases, &grad_out_b, lr_s);
                sgd_update(&mut self.fc2_weights, &grad_fc2_w, lr_s);
                sgd_update(&mut self.fc2_biases, &grad_fc2_b, lr_s);
                sgd_update(&mut self.fc1_weights, &grad_fc1_w, lr_s);
                sgd_update(&mut self.fc1_biases, &grad_fc1_b, lr_s);
                sgd_update(&mut self.conv2_weights, &grad_conv2_w, lr_s);
                sgd_update(&mut self.conv2_biases, &grad_conv2_b, lr_s);
                sgd_update(&mut self.conv1_weights, &grad_conv1_w, lr_s);
                sgd_update(&mut self.conv1_biases, &grad_conv1_b, lr_s);
            }

            println!("  Epoch {:4}  loss: {:.4}", epoch, epoch_loss / num_samples as f32);
        }
    }

    fn evaluate(&self, inputs: &[f32], targets: &[i32], num_samples: usize) -> (f32, f32) {
        let eval_bs = 256;
        let mut ws_all = vec![0.0f32; eval_bs * WS_SIZE];
        let mut flat = vec![0.0f32; eval_bs * FLAT_SIZE];
        let mut fc1_out = vec![0.0f32; eval_bs * FC1_OUT];
        let mut fc2_out = vec![0.0f32; eval_bs * FC2_OUT];
        let mut out = vec![0.0f32; eval_bs * NUM_CLASSES];

        let input_size = IN_C * IN_H * IN_W;
        let num_batches = (num_samples + eval_bs - 1) / eval_bs;

        let mut total_loss = 0.0f32;
        let mut correct = 0;

        for batch in 0..num_batches {
            let start = batch * eval_bs;
            let bs = eval_bs.min(num_samples - start);

            self.forward_batch(&inputs[start * input_size..], bs,
                               &mut ws_all, &mut flat,
                               &mut fc1_out, &mut fc2_out, &mut out);

            for i in 0..bs {
                let o = &out[i * NUM_CLASSES..(i + 1) * NUM_CLASSES];
                let label = targets[start + i] as usize;
                total_loss -= (o[label] + 1e-7).ln();

                let mut pred = 0;
                let mut mx = o[0];
                for j in 1..NUM_CLASSES {
                    if o[j] > mx { mx = o[j]; pred = j; }
                }
                if pred == label { correct += 1; }
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
    init_thread_pool();

    let args = parse_args();

    if args.dataset != "mnist" {
        eprintln!("CNN only supports MNIST dataset. Got: {}", args.dataset);
        std::process::exit(1);
    }

    let mut dataset = load_dataset(&args);

    println!(
        "Dataset: {}  ({} samples, {} features, {} classes)",
        args.dataset, dataset.num_samples, dataset.input_size, dataset.output_size
    );

    // No z-score normalization for MNIST
    shuffle_data(
        &mut dataset.inputs,
        &mut dataset.labels,
        dataset.num_samples,
        dataset.input_size,
    );

    let train_size = (dataset.num_samples as f32 * 0.8) as usize;
    let test_size = dataset.num_samples - train_size;
    println!("Train: {} samples, Test: {} samples", train_size, test_size);

    let mut cnn = Cnn::new();

    let num_epochs = args.epochs;
    let batch_size = args.batch_size;

    println!(
        "\nTraining LeNet-5 ({} epochs, batch_size={}, lr={:.4})...",
        num_epochs, batch_size, LEARNING_RATE
    );

    let t_start = Instant::now();
    cnn.train(
        &dataset.inputs[..train_size * dataset.input_size],
        &dataset.labels[..train_size],
        train_size,
        batch_size,
        num_epochs,
        LEARNING_RATE,
    );
    let t_train = t_start.elapsed().as_secs_f64();

    let t_eval_start = Instant::now();
    let (loss, accuracy) = cnn.evaluate(
        &dataset.inputs[train_size * dataset.input_size..],
        &dataset.labels[train_size..],
        test_size,
    );
    let t_eval = t_eval_start.elapsed().as_secs_f64();

    println!("\n=== Results on {} ===", args.dataset);
    println!("Test Loss:     {:.4}", loss);
    println!("Test Accuracy: {:.2}%", accuracy);
    println!("Train time:    {:.3} s", t_train);
    println!("Eval time:     {:.3} s", t_eval);
    println!(
        "Throughput:    {:.0} samples/s",
        train_size as f64 * num_epochs as f64 / t_train
    );
}
