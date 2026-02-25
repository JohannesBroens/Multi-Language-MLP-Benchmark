use nn_common::{load_dataset, parse_args, shuffle_data};
use std::time::Instant;

const LEARNING_RATE: f32 = 0.01;

// LeNet-5 dimensions
const IN_C: usize = 1;
const IN_H: usize = 28;
const IN_W: usize = 28;
const C1_OUT: i32 = 6;
const C1_K: i32 = 5;
const C1_OH: i32 = 24;
const C1_OW: i32 = 24;
const C1_COL_ROWS: i32 = 25;  // 1*5*5
const C1_COL_COLS: i32 = 576;  // 24*24
const P1_SIZE: i32 = 2;
const P1_H: i32 = 12;
const P1_W: i32 = 12;
const C2_OUT: i32 = 16;
const C2_IN: i32 = 6;
const C2_K: i32 = 5;
const C2_OH: i32 = 8;
const C2_OW: i32 = 8;
const C2_COL_ROWS: i32 = 150;  // 6*5*5
const C2_COL_COLS: i32 = 64;   // 8*8
const P2_SIZE: i32 = 2;
const FLAT_SIZE: i32 = 256;    // 16*4*4
const FC1_OUT: i32 = 120;
const FC2_OUT: i32 = 84;
const NUM_CLASSES: i32 = 10;

// Workspace offsets
const WS_COL1: i32 = 0;
const WS_CONV1: i32 = C1_COL_ROWS * C1_COL_COLS;
const WS_POOL1: i32 = WS_CONV1 + C1_OUT * C1_COL_COLS;
const WS_COL2: i32 = WS_POOL1 + C1_OUT * P1_H * P1_W;
const WS_CONV2: i32 = WS_COL2 + C2_COL_ROWS * C2_COL_COLS;
const WS_POOL2: i32 = WS_CONV2 + C2_OUT * C2_COL_COLS;
const WS_SIZE: i32 = WS_POOL2 + FLAT_SIZE;

// ---------------------------------------------------------------------------
//  CUDA FFI
// ---------------------------------------------------------------------------

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;
const CUDA_MANAGED_MEM_GLOBAL: u32 = 1;

extern "C" {
    fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: i32) -> i32;
    fn cudaMemset(devPtr: *mut std::ffi::c_void, value: i32, count: usize) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaMallocManaged(devPtr: *mut *mut std::ffi::c_void, size: usize, flags: u32) -> i32;

    // Custom tiled matmul kernels (NO cuBLAS)
    fn launch_matmul_nn(A: *const f32, B: *const f32, C: *mut f32, M: i32, N: i32, K: i32);
    fn launch_matmul_tn(A: *const f32, B: *const f32, C: *mut f32, M: i32, N: i32, K: i32);
    fn launch_matmul_nt(A: *const f32, B: *const f32, C: *mut f32, M: i32, N: i32, K: i32);

    // CNN kernels
    fn launch_im2col(input: *const f32, col: *mut f32, C: i32, H: i32, W: i32, kH: i32, kW: i32, stride: i32, OH: i32, OW: i32);
    fn launch_col2im(col: *const f32, output: *mut f32, C: i32, H: i32, W: i32, kH: i32, kW: i32, stride: i32, OH: i32, OW: i32);
    fn launch_avg_pool_forward(input: *const f32, output: *mut f32, C: i32, H: i32, W: i32, pool_size: i32, OH: i32, OW: i32);
    fn launch_avg_pool_backward(grad_output: *const f32, grad_input: *mut f32, C: i32, H: i32, W: i32, pool_size: i32, OH: i32, OW: i32);
    fn launch_add_bias(x: *mut f32, bias: *const f32, channels: i32, spatial: i32);
    fn launch_bias_grad(grad: *const f32, grad_bias: *mut f32, channels: i32, spatial: i32);
    fn launch_copy_flat(ws_all: *const f32, flat: *mut f32, ws_size: i32, ws_pool2_offset: i32, flat_size: i32, bs: i32);

    // Shared elementwise kernels
    fn launch_bias_relu(hidden: *mut f32, bias: *const f32, total: i32, hid: i32);
    fn launch_relu_forward(x: *mut f32, n: i32);
    fn launch_bias_softmax(output: *mut f32, bias: *const f32, bs: i32, out: i32);
    fn launch_ce_grad(output: *const f32, targets: *const i32, d_output: *mut f32, losses: *mut f32, bs: i32, out: i32);
    fn launch_relu_mask(d_hidden: *mut f32, hidden: *const f32, total: i32);
    fn launch_col_sum(mat: *const f32, out_vec: *mut f32, rows: i32, cols: i32);
    fn launch_sgd(param: *mut f32, grad: *const f32, lr: f32, n: i32);
    fn launch_reduce_sum(arr: *const f32, result: *mut f32, n: i32);
    fn launch_init_weights(weights: *mut f32, size: i32, scale: f32, seed: u64);
    fn launch_eval_metrics(output: *const f32, targets: *const i32, losses: *mut f32, correct: *mut i32, bs: i32, out: i32);
}

// ---------------------------------------------------------------------------
//  CUDA helpers
// ---------------------------------------------------------------------------

fn cuda_malloc(size: usize) -> *mut f32 {
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let err = unsafe { cudaMalloc(&mut ptr, size) };
    if err != 0 { panic!("cudaMalloc failed: {}", err); }
    ptr as *mut f32
}

fn cuda_malloc_managed(size: usize) -> *mut f32 {
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let err = unsafe { cudaMallocManaged(&mut ptr, size, CUDA_MANAGED_MEM_GLOBAL) };
    if err != 0 { panic!("cudaMallocManaged failed: {}", err); }
    ptr as *mut f32
}

fn cuda_free(ptr: *mut f32) { unsafe { cudaFree(ptr as *mut std::ffi::c_void); } }
fn cuda_sync() { let err = unsafe { cudaDeviceSynchronize() }; if err != 0 { panic!("sync failed: {}", err); } }

fn cuda_memcpy_h2d(dst: *mut f32, src: &[f32]) {
    unsafe { cudaMemcpy(dst as *mut _, src.as_ptr() as *const _, src.len() * 4, CUDA_MEMCPY_HOST_TO_DEVICE); }
}
fn cuda_memcpy_h2d_i32(dst: *mut i32, src: &[i32]) {
    unsafe { cudaMemcpy(dst as *mut _, src.as_ptr() as *const _, src.len() * 4, CUDA_MEMCPY_HOST_TO_DEVICE); }
}
fn cuda_memcpy_d2d(dst: *mut f32, src: *const f32, count: usize) {
    unsafe { cudaMemcpy(dst as *mut _, src as *const _, count * 4, CUDA_MEMCPY_DEVICE_TO_DEVICE); }
}
fn cuda_memset_zero(ptr: *mut f32, count: usize) {
    unsafe { cudaMemset(ptr as *mut _, 0, count * 4); }
}

// ---------------------------------------------------------------------------
//  CNN struct
// ---------------------------------------------------------------------------

struct GpuCnn {
    conv1_w: *mut f32, conv1_b: *mut f32,
    conv2_w: *mut f32, conv2_b: *mut f32,
    fc1_w: *mut f32, fc1_b: *mut f32,
    fc2_w: *mut f32, fc2_b: *mut f32,
    out_w: *mut f32, out_b: *mut f32,
}

impl GpuCnn {
    fn new() -> Self {
        let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();

        let conv1_w = cuda_malloc_managed((C1_OUT * C1_COL_ROWS) as usize * 4);
        let conv1_b = cuda_malloc_managed(C1_OUT as usize * 4);
        let conv2_w = cuda_malloc_managed((C2_OUT * C2_COL_ROWS) as usize * 4);
        let conv2_b = cuda_malloc_managed(C2_OUT as usize * 4);
        let fc1_w = cuda_malloc_managed((FLAT_SIZE * FC1_OUT) as usize * 4);
        let fc1_b = cuda_malloc_managed(FC1_OUT as usize * 4);
        let fc2_w = cuda_malloc_managed((FC1_OUT * FC2_OUT) as usize * 4);
        let fc2_b = cuda_malloc_managed(FC2_OUT as usize * 4);
        let out_w = cuda_malloc_managed((FC2_OUT * NUM_CLASSES) as usize * 4);
        let out_b = cuda_malloc_managed(NUM_CLASSES as usize * 4);

        cuda_memset_zero(conv1_b, C1_OUT as usize);
        cuda_memset_zero(conv2_b, C2_OUT as usize);
        cuda_memset_zero(fc1_b, FC1_OUT as usize);
        cuda_memset_zero(fc2_b, FC2_OUT as usize);
        cuda_memset_zero(out_b, NUM_CLASSES as usize);

        unsafe {
            launch_init_weights(conv1_w, C1_OUT * C1_COL_ROWS, (2.0f32 / C1_COL_ROWS as f32).sqrt(), seed);
            launch_init_weights(conv2_w, C2_OUT * C2_COL_ROWS, (2.0f32 / C2_COL_ROWS as f32).sqrt(), seed + 1);
            launch_init_weights(fc1_w, FLAT_SIZE * FC1_OUT, (2.0f32 / FLAT_SIZE as f32).sqrt(), seed + 2);
            launch_init_weights(fc2_w, FC1_OUT * FC2_OUT, (2.0f32 / FC1_OUT as f32).sqrt(), seed + 3);
            launch_init_weights(out_w, FC2_OUT * NUM_CLASSES, (2.0f32 / FC2_OUT as f32).sqrt(), seed + 4);
        }
        cuda_sync();

        GpuCnn { conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, out_w, out_b }
    }

    fn conv_forward_sample(&self, input: *const f32, ws: *mut f32) {
        unsafe {
            let col1 = ws.add(WS_COL1 as usize);
            let conv1 = ws.add(WS_CONV1 as usize);
            let pool1 = ws.add(WS_POOL1 as usize);
            let col2 = ws.add(WS_COL2 as usize);
            let conv2 = ws.add(WS_CONV2 as usize);
            let pool2 = ws.add(WS_POOL2 as usize);

            launch_im2col(input, col1, 1, IN_H as i32, IN_W as i32, C1_K, C1_K, 1, C1_OH, C1_OW);
            launch_matmul_nn(self.conv1_w, col1, conv1, C1_OUT, C1_COL_COLS, C1_COL_ROWS);
            launch_add_bias(conv1, self.conv1_b, C1_OUT, C1_COL_COLS);
            launch_relu_forward(conv1, C1_OUT * C1_COL_COLS);
            launch_avg_pool_forward(conv1, pool1, C1_OUT, C1_OH, C1_OW, P1_SIZE, P1_H, P1_W);

            launch_im2col(pool1, col2, C2_IN, P1_H, P1_W, C2_K, C2_K, 1, C2_OH, C2_OW);
            launch_matmul_nn(self.conv2_w, col2, conv2, C2_OUT, C2_COL_COLS, C2_COL_ROWS);
            launch_add_bias(conv2, self.conv2_b, C2_OUT, C2_COL_COLS);
            launch_relu_forward(conv2, C2_OUT * C2_COL_COLS);
            launch_avg_pool_forward(conv2, pool2, C2_OUT, C2_OH, C2_OW, P2_SIZE, C2_OH / P2_SIZE, C2_OW / P2_SIZE);
        }
    }

    fn forward_batch(&self, d_inputs: *const f32, bs: i32,
                     ws_all: *mut f32, flat: *mut f32,
                     fc1_out: *mut f32, fc2_out: *mut f32, out: *mut f32) {
        let input_size = (IN_C * IN_H * IN_W) as usize;
        for b in 0..bs as usize {
            let img = unsafe { d_inputs.add(b * input_size) };
            let ws = unsafe { ws_all.add(b * WS_SIZE as usize) };
            self.conv_forward_sample(img, ws);
        }
        unsafe { launch_copy_flat(ws_all, flat, WS_SIZE, WS_POOL2, FLAT_SIZE, bs); }

        unsafe { launch_matmul_nn(flat, self.fc1_w, fc1_out, bs, FC1_OUT, FLAT_SIZE); }
        unsafe { launch_bias_relu(fc1_out, self.fc1_b, bs * FC1_OUT, FC1_OUT); }

        unsafe { launch_matmul_nn(fc1_out, self.fc2_w, fc2_out, bs, FC2_OUT, FC1_OUT); }
        unsafe { launch_bias_relu(fc2_out, self.fc2_b, bs * FC2_OUT, FC2_OUT); }

        unsafe { launch_matmul_nn(fc2_out, self.out_w, out, bs, NUM_CLASSES, FC2_OUT); }
        unsafe { launch_bias_softmax(out, self.out_b, bs, NUM_CLASSES); }
    }

    fn train(&mut self, inputs: &[f32], targets: &[i32], num_samples: usize,
             batch_size: usize, num_epochs: usize) {
        let input_size = IN_C * IN_H * IN_W;
        let d_inputs = cuda_malloc(num_samples * input_size * 4);
        let d_targets = cuda_malloc(num_samples * 4) as *mut i32;
        cuda_memcpy_h2d(d_inputs, &inputs[..num_samples * input_size]);
        cuda_memcpy_h2d_i32(d_targets, &targets[..num_samples]);

        let ws_all = cuda_malloc(batch_size * WS_SIZE as usize * 4);
        let flat = cuda_malloc(batch_size * FLAT_SIZE as usize * 4);
        let fc1_out = cuda_malloc(batch_size * FC1_OUT as usize * 4);
        let fc2_out = cuda_malloc(batch_size * FC2_OUT as usize * 4);
        let out = cuda_malloc(batch_size * NUM_CLASSES as usize * 4);

        let d_out = cuda_malloc(batch_size * NUM_CLASSES as usize * 4);
        let d_fc2 = cuda_malloc(batch_size * FC2_OUT as usize * 4);
        let d_fc1 = cuda_malloc(batch_size * FC1_OUT as usize * 4);
        let d_flat = cuda_malloc(batch_size * FLAT_SIZE as usize * 4);

        let g_out_w = cuda_malloc(FC2_OUT as usize * NUM_CLASSES as usize * 4);
        let g_out_b = cuda_malloc(NUM_CLASSES as usize * 4);
        let g_fc2_w = cuda_malloc(FC1_OUT as usize * FC2_OUT as usize * 4);
        let g_fc2_b = cuda_malloc(FC2_OUT as usize * 4);
        let g_fc1_w = cuda_malloc(FLAT_SIZE as usize * FC1_OUT as usize * 4);
        let g_fc1_b = cuda_malloc(FC1_OUT as usize * 4);

        let g_conv2_w = cuda_malloc(C2_OUT as usize * C2_COL_ROWS as usize * 4);
        let g_conv2_b = cuda_malloc(C2_OUT as usize * 4);
        let g_conv1_w = cuda_malloc(C1_OUT as usize * C1_COL_ROWS as usize * 4);
        let g_conv1_b = cuda_malloc(C1_OUT as usize * 4);

        let d_pool2 = cuda_malloc(FLAT_SIZE as usize * 4);
        let d_conv2_out = cuda_malloc((C2_OUT * C2_COL_COLS) as usize * 4);
        let d_col2 = cuda_malloc((C2_COL_ROWS * C2_COL_COLS) as usize * 4);
        let d_pool1 = cuda_malloc((C2_IN * P1_H * P1_W) as usize * 4);
        let d_conv1_out = cuda_malloc((C1_OUT * C1_COL_COLS) as usize * 4);

        let tmp_gw2 = cuda_malloc((C2_OUT * C2_COL_ROWS) as usize * 4);
        let tmp_gw1 = cuda_malloc((C1_OUT * C1_COL_ROWS) as usize * 4);
        let tmp_gb2 = cuda_malloc(C2_OUT as usize * 4);
        let tmp_gb1 = cuda_malloc(C1_OUT as usize * 4);

        let d_losses = cuda_malloc(batch_size * 4);
        let d_loss_sum = cuda_malloc_managed(4);

        let num_batches = (num_samples + batch_size - 1) / batch_size;

        for epoch in 0..num_epochs {
            unsafe { *d_loss_sum = 0.0f32; }

            for batch in 0..num_batches {
                let start = batch * batch_size;
                let bs = batch_size.min(num_samples - start) as i32;

                let x = unsafe { d_inputs.add(start * input_size) };
                let y = unsafe { d_targets.add(start) };

                self.forward_batch(x, bs, ws_all, flat, fc1_out, fc2_out, out);

                unsafe {
                    launch_ce_grad(out, y, d_out, d_losses, bs, NUM_CLASSES);
                    launch_reduce_sum(d_losses, d_loss_sum, bs);
                }

                // FC backward
                unsafe { launch_matmul_tn(fc2_out, d_out, g_out_w, FC2_OUT, NUM_CLASSES, bs); }
                unsafe { launch_col_sum(d_out, g_out_b, bs, NUM_CLASSES); }

                unsafe { launch_matmul_nt(d_out, self.out_w, d_fc2, bs, FC2_OUT, NUM_CLASSES); }
                unsafe { launch_relu_mask(d_fc2, fc2_out, bs * FC2_OUT); }

                unsafe { launch_matmul_tn(fc1_out, d_fc2, g_fc2_w, FC1_OUT, FC2_OUT, bs); }
                unsafe { launch_col_sum(d_fc2, g_fc2_b, bs, FC2_OUT); }

                unsafe { launch_matmul_nt(d_fc2, self.fc2_w, d_fc1, bs, FC1_OUT, FC2_OUT); }
                unsafe { launch_relu_mask(d_fc1, fc1_out, bs * FC1_OUT); }

                unsafe { launch_matmul_tn(flat, d_fc1, g_fc1_w, FLAT_SIZE, FC1_OUT, bs); }
                unsafe { launch_col_sum(d_fc1, g_fc1_b, bs, FC1_OUT); }

                unsafe { launch_matmul_nt(d_fc1, self.fc1_w, d_flat, bs, FLAT_SIZE, FC1_OUT); }

                // Conv backward (per-sample)
                cuda_memset_zero(g_conv2_w, (C2_OUT * C2_COL_ROWS) as usize);
                cuda_memset_zero(g_conv2_b, C2_OUT as usize);
                cuda_memset_zero(g_conv1_w, (C1_OUT * C1_COL_ROWS) as usize);
                cuda_memset_zero(g_conv1_b, C1_OUT as usize);

                for b in 0..bs {
                    let ws = unsafe { ws_all.add(b as usize * WS_SIZE as usize) };
                    let col1 = unsafe { ws.add(WS_COL1 as usize) };
                    let conv1_out_ptr = unsafe { ws.add(WS_CONV1 as usize) };
                    let col2 = unsafe { ws.add(WS_COL2 as usize) };
                    let conv2_out_ptr = unsafe { ws.add(WS_CONV2 as usize) };

                    cuda_memcpy_d2d(d_pool2, unsafe { d_flat.add(b as usize * FLAT_SIZE as usize) }, FLAT_SIZE as usize);

                    cuda_memset_zero(d_conv2_out, (C2_OUT * C2_COL_COLS) as usize);
                    unsafe { launch_avg_pool_backward(d_pool2, d_conv2_out, C2_OUT, C2_OH, C2_OW, P2_SIZE, C2_OH / P2_SIZE, C2_OW / P2_SIZE); }
                    unsafe { launch_relu_mask(d_conv2_out, conv2_out_ptr, C2_OUT * C2_COL_COLS); }

                    unsafe { launch_matmul_nt(d_conv2_out, col2, tmp_gw2, C2_OUT, C2_COL_ROWS, C2_COL_COLS); }
                    unsafe { launch_sgd(g_conv2_w, tmp_gw2, -1.0, C2_OUT * C2_COL_ROWS); }
                    unsafe { launch_bias_grad(d_conv2_out, tmp_gb2, C2_OUT, C2_COL_COLS); }
                    unsafe { launch_sgd(g_conv2_b, tmp_gb2, -1.0, C2_OUT); }

                    unsafe { launch_matmul_tn(self.conv2_w, d_conv2_out, d_col2, C2_COL_ROWS, C2_COL_COLS, C2_OUT); }

                    cuda_memset_zero(d_pool1, (C2_IN * P1_H * P1_W) as usize);
                    unsafe { launch_col2im(d_col2, d_pool1, C2_IN, P1_H, P1_W, C2_K, C2_K, 1, C2_OH, C2_OW); }

                    cuda_memset_zero(d_conv1_out, (C1_OUT * C1_COL_COLS) as usize);
                    unsafe { launch_avg_pool_backward(d_pool1, d_conv1_out, C1_OUT, C1_OH, C1_OW, P1_SIZE, P1_H, P1_W); }
                    unsafe { launch_relu_mask(d_conv1_out, conv1_out_ptr, C1_OUT * C1_COL_COLS); }

                    unsafe { launch_matmul_nt(d_conv1_out, col1, tmp_gw1, C1_OUT, C1_COL_ROWS, C1_COL_COLS); }
                    unsafe { launch_sgd(g_conv1_w, tmp_gw1, -1.0, C1_OUT * C1_COL_ROWS); }
                    unsafe { launch_bias_grad(d_conv1_out, tmp_gb1, C1_OUT, C1_COL_COLS); }
                    unsafe { launch_sgd(g_conv1_b, tmp_gb1, -1.0, C1_OUT); }
                }

                let lr_s = LEARNING_RATE / bs as f32;
                unsafe {
                    launch_sgd(self.out_w, g_out_w, lr_s, FC2_OUT * NUM_CLASSES);
                    launch_sgd(self.out_b, g_out_b, lr_s, NUM_CLASSES);
                    launch_sgd(self.fc2_w, g_fc2_w, lr_s, FC1_OUT * FC2_OUT);
                    launch_sgd(self.fc2_b, g_fc2_b, lr_s, FC2_OUT);
                    launch_sgd(self.fc1_w, g_fc1_w, lr_s, FLAT_SIZE * FC1_OUT);
                    launch_sgd(self.fc1_b, g_fc1_b, lr_s, FC1_OUT);
                    launch_sgd(self.conv2_w, g_conv2_w, lr_s, C2_OUT * C2_COL_ROWS);
                    launch_sgd(self.conv2_b, g_conv2_b, lr_s, C2_OUT);
                    launch_sgd(self.conv1_w, g_conv1_w, lr_s, C1_OUT * C1_COL_ROWS);
                    launch_sgd(self.conv1_b, g_conv1_b, lr_s, C1_OUT);
                }
            }

            cuda_sync();
            println!("  Epoch {:4}  loss: {:.4}", epoch, unsafe { *d_loss_sum } / num_samples as f32);
        }

        cuda_sync();
        for p in [d_inputs, d_targets as *mut f32, ws_all, flat, fc1_out, fc2_out, out,
                   d_out, d_fc2, d_fc1, d_flat, g_out_w, g_out_b, g_fc2_w, g_fc2_b,
                   g_fc1_w, g_fc1_b, g_conv2_w, g_conv2_b, g_conv1_w, g_conv1_b,
                   d_pool2, d_conv2_out, d_col2, d_pool1, d_conv1_out,
                   tmp_gw2, tmp_gw1, tmp_gb2, tmp_gb1, d_losses, d_loss_sum] {
            cuda_free(p);
        }
    }

    fn evaluate(&self, inputs: &[f32], targets: &[i32], num_samples: usize) -> (f32, f32) {
        let input_size = IN_C * IN_H * IN_W;
        let eval_bs = 256i32;

        let d_inputs = cuda_malloc(num_samples * input_size * 4);
        let d_targets = cuda_malloc(num_samples * 4) as *mut i32;
        cuda_memcpy_h2d(d_inputs, &inputs[..num_samples * input_size]);
        cuda_memcpy_h2d_i32(d_targets, &targets[..num_samples]);

        let ws_all = cuda_malloc(eval_bs as usize * WS_SIZE as usize * 4);
        let flat = cuda_malloc(eval_bs as usize * FLAT_SIZE as usize * 4);
        let fc1_out = cuda_malloc(eval_bs as usize * FC1_OUT as usize * 4);
        let fc2_out = cuda_malloc(eval_bs as usize * FC2_OUT as usize * 4);
        let out = cuda_malloc(eval_bs as usize * NUM_CLASSES as usize * 4);
        let d_losses = cuda_malloc(eval_bs as usize * 4);
        let d_loss_sum = cuda_malloc_managed(4);
        let d_correct = cuda_malloc_managed(4) as *mut i32;

        unsafe { *d_loss_sum = 0.0f32; *d_correct = 0i32; }

        let num_batches = (num_samples + eval_bs as usize - 1) / eval_bs as usize;
        for batch in 0..num_batches {
            let start = batch * eval_bs as usize;
            let bs = (eval_bs as usize).min(num_samples - start) as i32;

            self.forward_batch(unsafe { d_inputs.add(start * input_size) }, bs,
                               ws_all, flat, fc1_out, fc2_out, out);

            unsafe {
                launch_eval_metrics(out, d_targets.add(start), d_losses, d_correct, bs, NUM_CLASSES);
                launch_reduce_sum(d_losses, d_loss_sum, bs);
            }
        }

        cuda_sync();
        let loss = unsafe { *d_loss_sum } / num_samples as f32;
        let accuracy = unsafe { *d_correct } as f32 / num_samples as f32 * 100.0;

        for p in [d_inputs, d_targets as *mut f32, ws_all, flat, fc1_out, fc2_out, out, d_losses, d_loss_sum, d_correct as *mut f32] {
            cuda_free(p);
        }
        (loss, accuracy)
    }
}

impl Drop for GpuCnn {
    fn drop(&mut self) {
        for p in [self.conv1_w, self.conv1_b, self.conv2_w, self.conv2_b,
                   self.fc1_w, self.fc1_b, self.fc2_w, self.fc2_b,
                   self.out_w, self.out_b] {
            cuda_free(p);
        }
    }
}

fn main() {
    let args = parse_args();
    if args.dataset != "mnist" {
        eprintln!("CNN only supports MNIST dataset. Got: {}", args.dataset);
        std::process::exit(1);
    }

    let mut dataset = load_dataset(&args);
    println!("Dataset: {}  ({} samples, {} features, {} classes)",
             args.dataset, dataset.num_samples, dataset.input_size, dataset.output_size);

    shuffle_data(&mut dataset.inputs, &mut dataset.labels, dataset.num_samples, dataset.input_size);

    let train_size = (dataset.num_samples as f32 * 0.8) as usize;
    let test_size = dataset.num_samples - train_size;
    println!("Train: {} samples, Test: {} samples", train_size, test_size);

    let mut cnn = GpuCnn::new();
    println!("\nTraining LeNet-5 ({} epochs, batch_size={}, lr={:.4})...",
             args.epochs, args.batch_size, LEARNING_RATE);

    let t_start = Instant::now();
    cnn.train(&dataset.inputs[..train_size * dataset.input_size],
              &dataset.labels[..train_size], train_size, args.batch_size, args.epochs);
    let t_train = t_start.elapsed().as_secs_f64();

    let t_eval_start = Instant::now();
    let (loss, accuracy) = cnn.evaluate(
        &dataset.inputs[train_size * dataset.input_size..],
        &dataset.labels[train_size..], test_size);
    let t_eval = t_eval_start.elapsed().as_secs_f64();

    println!("\n=== Results on {} ===", args.dataset);
    println!("Test Loss:     {:.4}", loss);
    println!("Test Accuracy: {:.2}%", accuracy);
    println!("Train time:    {:.3} s", t_train);
    println!("Eval time:     {:.3} s", t_eval);
    println!("Throughput:    {:.0} samples/s", train_size as f64 * args.epochs as f64 / t_train);
}
