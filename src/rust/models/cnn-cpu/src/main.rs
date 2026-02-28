use nn_common::{
    parse_args, load_dataset, shuffle_data, Xorshift32,
    sgemm_nn, sgemm_tn, sgemm_nt, init_thread_pool,
    bias_relu, bias_softmax, cross_entropy_grad, col_sum, sgd_update,
    relu_forward, relu_backward, xavier_init,
};
use std::time::Instant;

const DEFAULT_LR: f32 = 0.32;

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
//  Per-sample im2col / col2im / pooling (kept for compatibility)
// ---------------------------------------------------------------------------

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
//  Batched helpers: channel-major [C, B*spatial] layout
// ---------------------------------------------------------------------------

/// Batched im2col: input [C, B*H*W] -> col [K, B*OH*OW]
fn im2col_batch(input: &[f32], batch: usize, c: usize, h: usize, w: usize,
                kh: usize, kw: usize, stride: usize, col: &mut [f32]) {
    let oh = (h - kh) / stride + 1;
    let ow = (w - kw) / stride + 1;
    let spatial_out = oh * ow;
    let col_cols = batch * spatial_out;

    let mut col_row = 0;
    for ci in 0..c {
        for khi in 0..kh {
            for kwi in 0..kw {
                let col_ptr = col_row * col_cols;
                for b in 0..batch {
                    let in_base = ci * (batch * h * w) + b * (h * w);
                    let out_base = col_ptr + b * spatial_out;
                    for ohi in 0..oh {
                        for owi in 0..ow {
                            let hi = ohi * stride + khi;
                            let wi = owi * stride + kwi;
                            col[out_base + ohi * ow + owi] =
                                input[in_base + hi * w + wi];
                        }
                    }
                }
                col_row += 1;
            }
        }
    }
}

/// Batched col2im: col [K, B*OH*OW] -> output [C, B*H*W] (accumulates)
fn col2im_batch(col: &[f32], batch: usize, c: usize, h: usize, w: usize,
                kh: usize, kw: usize, stride: usize, output: &mut [f32]) {
    let oh = (h - kh) / stride + 1;
    let ow = (w - kw) / stride + 1;
    let spatial_out = oh * ow;
    let col_cols = batch * spatial_out;

    let mut col_row = 0;
    for ci in 0..c {
        for khi in 0..kh {
            for kwi in 0..kw {
                let col_ptr = col_row * col_cols;
                for b in 0..batch {
                    let out_base = ci * (batch * h * w) + b * (h * w);
                    let in_base = col_ptr + b * spatial_out;
                    for ohi in 0..oh {
                        for owi in 0..ow {
                            let hi = ohi * stride + khi;
                            let wi = owi * stride + kwi;
                            output[out_base + hi * w + wi] +=
                                col[in_base + ohi * ow + owi];
                        }
                    }
                }
                col_row += 1;
            }
        }
    }
}

/// Batched avg pool forward: input [C, B*H*W] -> output [C, B*PH*PW]
fn avg_pool_forward_batch(input: &[f32], output: &mut [f32],
                          channels: usize, batch: usize,
                          h: usize, w: usize, pool_size: usize) {
    let ph = h / pool_size;
    let pw = w / pool_size;
    let scale = 1.0 / (pool_size * pool_size) as f32;

    for c in 0..channels {
        for b in 0..batch {
            let in_base = c * batch * h * w + b * h * w;
            let out_base = c * batch * ph * pw + b * ph * pw;
            for phi in 0..ph {
                for pwi in 0..pw {
                    let mut sum = 0.0f32;
                    for i in 0..pool_size {
                        for j in 0..pool_size {
                            sum += input[in_base + (phi * pool_size + i) * w + (pwi * pool_size + j)];
                        }
                    }
                    output[out_base + phi * pw + pwi] = sum * scale;
                }
            }
        }
    }
}

/// Batched avg pool backward: grad_output [C, B*PH*PW] -> grad_input [C, B*H*W]
fn avg_pool_backward_batch(grad_output: &[f32], grad_input: &mut [f32],
                           channels: usize, batch: usize,
                           h: usize, w: usize, pool_size: usize) {
    let ph = h / pool_size;
    let pw = w / pool_size;
    let scale = 1.0 / (pool_size * pool_size) as f32;

    for c in 0..channels {
        for b in 0..batch {
            let go_base = c * batch * ph * pw + b * ph * pw;
            let gi_base = c * batch * h * w + b * h * w;
            for phi in 0..ph {
                for pwi in 0..pw {
                    let g = grad_output[go_base + phi * pw + pwi] * scale;
                    for i in 0..pool_size {
                        for j in 0..pool_size {
                            grad_input[gi_base + (phi * pool_size + i) * w + (pwi * pool_size + j)] = g;
                        }
                    }
                }
            }
        }
    }
}

/// Transpose [C, B*S] -> [B, C*S]
fn transpose_cb_to_bc(src: &[f32], dst: &mut [f32], channels: usize, batch: usize, s: usize) {
    for b in 0..batch {
        for c in 0..channels {
            let src_off = c * batch * s + b * s;
            let dst_off = b * channels * s + c * s;
            dst[dst_off..dst_off + s].copy_from_slice(&src[src_off..src_off + s]);
        }
    }
}

/// Transpose [B, C*S] -> [C, B*S]
fn transpose_bc_to_cb(src: &[f32], dst: &mut [f32], batch: usize, channels: usize, s: usize) {
    for c in 0..channels {
        for b in 0..batch {
            let src_off = b * channels * s + c * s;
            let dst_off = c * batch * s + b * s;
            dst[dst_off..dst_off + s].copy_from_slice(&src[src_off..src_off + s]);
        }
    }
}

/// Conv bias gradient: sum each row of [channels, spatial] -> [channels]
fn conv_bias_grad(grad: &[f32], grad_bias: &mut [f32], channels: usize, spatial: usize) {
    for c in 0..channels {
        let mut sum = 0.0f32;
        let base = c * spatial;
        for j in 0..spatial {
            sum += grad[base + j];
        }
        grad_bias[c] = sum;
    }
}

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

    /// Batched forward: one GEMM per conv layer, then batched FC.
    fn forward_batch(&self, inputs: &[f32], bs: usize,
                     col1_b: &mut [f32], conv1_b: &mut [f32], pool1_b: &mut [f32],
                     col2_b: &mut [f32], conv2_b: &mut [f32], pool2_b: &mut [f32],
                     flat: &mut [f32],
                     fc1_out: &mut [f32], fc2_out: &mut [f32], out: &mut [f32]) {
        // Conv1: inputs[1, B*784] -> col1_b[25, B*576] -> conv1_b[6, B*576]
        im2col_batch(inputs, bs, C1_IN, IN_H, IN_W, C1_K, C1_K, 1, col1_b);
        sgemm_nn(C1_OUT, bs * C1_COL_COLS, C1_COL_ROWS,
                 &self.conv1_weights, col1_b, conv1_b);
        add_bias(conv1_b, &self.conv1_biases, C1_OUT, bs * C1_COL_COLS);
        relu_forward(conv1_b);
        avg_pool_forward_batch(conv1_b, pool1_b, C1_OUT, bs, C1_OH, C1_OW, P1_SIZE);

        // Conv2: pool1_b[6, B*144] -> col2_b[150, B*64] -> conv2_b[16, B*64]
        im2col_batch(pool1_b, bs, C2_IN, P1_H, P1_W, C2_K, C2_K, 1, col2_b);
        sgemm_nn(C2_OUT, bs * C2_COL_COLS, C2_COL_ROWS,
                 &self.conv2_weights, col2_b, conv2_b);
        add_bias(conv2_b, &self.conv2_biases, C2_OUT, bs * C2_COL_COLS);
        relu_forward(conv2_b);
        avg_pool_forward_batch(conv2_b, pool2_b, C2_OUT, bs, C2_OH, C2_OW, P2_SIZE);

        // Transition: pool2_b[16, B*16] -> flat[B, 256]
        transpose_cb_to_bc(pool2_b, flat, C2_OUT, bs, P2_H * P2_W);

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
             batch_size: usize, num_epochs: usize, learning_rate: f32,
             optimizer: &str, scheduler: &str) {
        if optimizer == "adam" {
            eprintln!("Warning: Adam optimizer not yet implemented, using SGD");
        }
        let _ = scheduler; // will be used in future
        let input_size = IN_C * IN_H * IN_W;

        // Forward conv workspace (batched, channel-major)
        let mut col1_b = vec![0.0f32; C1_COL_ROWS * batch_size * C1_COL_COLS];
        let mut conv1_b = vec![0.0f32; C1_OUT * batch_size * C1_COL_COLS];
        let mut pool1_b = vec![0.0f32; C1_OUT * batch_size * P1_H * P1_W];
        let mut col2_b = vec![0.0f32; C2_COL_ROWS * batch_size * C2_COL_COLS];
        let mut conv2_b = vec![0.0f32; C2_OUT * batch_size * C2_COL_COLS];
        let mut pool2_b = vec![0.0f32; C2_OUT * batch_size * P2_H * P2_W];

        // FC buffers
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

        // Backward conv workspace (batched, channel-major)
        let mut d_pool2_b = vec![0.0f32; C2_OUT * batch_size * P2_H * P2_W];
        let mut d_conv2_b = vec![0.0f32; C2_OUT * batch_size * C2_COL_COLS];
        let mut d_col2_b = vec![0.0f32; C2_COL_ROWS * batch_size * C2_COL_COLS];
        let mut d_pool1_b = vec![0.0f32; C2_IN * batch_size * P1_H * P1_W];
        let mut d_conv1_b = vec![0.0f32; C1_OUT * batch_size * C1_COL_COLS];

        let num_batches = (num_samples + batch_size - 1) / batch_size;

        for epoch in 0..num_epochs {
            let mut epoch_loss = 0.0f32;

            for batch in 0..num_batches {
                let start = batch * batch_size;
                let bs = batch_size.min(num_samples - start);

                let x = &inputs[start * input_size..];
                let y = &targets[start..];

                // Forward (batched)
                self.forward_batch(x, bs,
                                   &mut col1_b, &mut conv1_b, &mut pool1_b,
                                   &mut col2_b, &mut conv2_b, &mut pool2_b,
                                   &mut flat, &mut fc1_out, &mut fc2_out, &mut out);

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

                // Conv backward (batched)

                // Transition: d_flat[B, 256] -> d_pool2_b[16, B*16]
                transpose_bc_to_cb(&d_flat, &mut d_pool2_b, bs, C2_OUT, P2_H * P2_W);

                // Pool2 backward
                avg_pool_backward_batch(&d_pool2_b, &mut d_conv2_b,
                                        C2_OUT, bs, C2_OH, C2_OW, P2_SIZE);

                // ReLU backward conv2
                {
                    let n = C2_OUT * bs * C2_COL_COLS;
                    relu_backward(&mut d_conv2_b[..n], &conv2_b[..n]);
                }

                // Conv2 weight gradient (one GEMM)
                sgemm_nt(C2_OUT, C2_COL_ROWS, bs * C2_COL_COLS,
                         &d_conv2_b, &col2_b, &mut grad_conv2_w);

                // Conv2 bias gradient
                conv_bias_grad(&d_conv2_b, &mut grad_conv2_b, C2_OUT, bs * C2_COL_COLS);

                // Conv2 input gradient
                sgemm_tn(C2_COL_ROWS, bs * C2_COL_COLS, C2_OUT,
                         &self.conv2_weights, &d_conv2_b, &mut d_col2_b);

                // col2im batch
                for v in d_pool1_b.iter_mut() { *v = 0.0; }
                col2im_batch(&d_col2_b, bs, C2_IN, P1_H, P1_W, C2_K, C2_K, 1, &mut d_pool1_b);

                // Pool1 backward
                avg_pool_backward_batch(&d_pool1_b, &mut d_conv1_b,
                                        C1_OUT, bs, C1_OH, C1_OW, P1_SIZE);

                // ReLU backward conv1
                {
                    let n = C1_OUT * bs * C1_COL_COLS;
                    relu_backward(&mut d_conv1_b[..n], &conv1_b[..n]);
                }

                // Conv1 weight gradient (one GEMM)
                sgemm_nt(C1_OUT, C1_COL_ROWS, bs * C1_COL_COLS,
                         &d_conv1_b, &col1_b, &mut grad_conv1_w);

                // Conv1 bias gradient
                conv_bias_grad(&d_conv1_b, &mut grad_conv1_b, C1_OUT, bs * C1_COL_COLS);

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

        let mut col1_b = vec![0.0f32; C1_COL_ROWS * eval_bs * C1_COL_COLS];
        let mut conv1_b = vec![0.0f32; C1_OUT * eval_bs * C1_COL_COLS];
        let mut pool1_b = vec![0.0f32; C1_OUT * eval_bs * P1_H * P1_W];
        let mut col2_b = vec![0.0f32; C2_COL_ROWS * eval_bs * C2_COL_COLS];
        let mut conv2_b = vec![0.0f32; C2_OUT * eval_bs * C2_COL_COLS];
        let mut pool2_b = vec![0.0f32; C2_OUT * eval_bs * P2_H * P2_W];
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
                               &mut col1_b, &mut conv1_b, &mut pool1_b,
                               &mut col2_b, &mut conv2_b, &mut pool2_b,
                               &mut flat, &mut fc1_out, &mut fc2_out, &mut out);

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
    let learning_rate = if args.learning_rate > 0.0 { args.learning_rate } else { DEFAULT_LR };

    println!(
        "\nTraining LeNet-5 ({} epochs, batch_size={}, lr={:.4})...",
        num_epochs, batch_size, learning_rate
    );

    let t_start = Instant::now();
    cnn.train(
        &dataset.inputs[..train_size * dataset.input_size],
        &dataset.labels[..train_size],
        train_size,
        batch_size,
        num_epochs,
        learning_rate,
        &args.optimizer,
        &args.scheduler,
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
