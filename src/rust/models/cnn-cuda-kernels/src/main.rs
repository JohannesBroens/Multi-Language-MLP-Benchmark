use nn_common::{load_dataset, parse_args, shuffle_data};
use std::time::Instant;

const DEFAULT_LR: f32 = 0.32;

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
const P2_H: i32 = C2_OH / P2_SIZE;  // 4
const P2_W: i32 = C2_OW / P2_SIZE;  // 4
const FLAT_SIZE: i32 = 256;    // 16*4*4
const FC1_OUT: i32 = 120;
const FC2_OUT: i32 = 84;
const NUM_CLASSES: i32 = 10;

// ---------------------------------------------------------------------------
//  CUDA FFI
// ---------------------------------------------------------------------------

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
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

    // Batched CNN kernels
    fn launch_im2col_batch(input: *const f32, col: *mut f32, C: i32, B: i32, H: i32, W: i32, kH: i32, kW: i32, stride: i32, OH: i32, OW: i32);
    fn launch_col2im_batch(col: *const f32, output: *mut f32, C: i32, B: i32, H: i32, W: i32, kH: i32, kW: i32, stride: i32, OH: i32, OW: i32);
    fn launch_avg_pool_forward_batch(input: *const f32, output: *mut f32, C: i32, B: i32, H: i32, W: i32, pool_size: i32, PH: i32, PW: i32);
    fn launch_avg_pool_backward_batch(grad_output: *const f32, grad_input: *mut f32, C: i32, B: i32, H: i32, W: i32, pool_size: i32, PH: i32, PW: i32);
    fn launch_transpose_cb_to_bc(src: *const f32, dst: *mut f32, C: i32, B: i32, S: i32);
    fn launch_transpose_bc_to_cb(src: *const f32, dst: *mut f32, B: i32, C: i32, S: i32);
    fn launch_add_bias(x: *mut f32, bias: *const f32, channels: i32, spatial: i32);
    fn launch_bias_grad(grad: *const f32, grad_bias: *mut f32, channels: i32, spatial: i32);

    // Shared elementwise kernels
    fn launch_bias_relu(hidden: *mut f32, bias: *const f32, total: i32, hid: i32);
    fn launch_relu_forward(x: *mut f32, n: i32);
    fn launch_bias_softmax(output: *mut f32, bias: *const f32, bs: i32, out: i32);
    fn launch_ce_grad(output: *const f32, targets: *const i32, d_output: *mut f32, losses: *mut f32, bs: i32, out: i32);
    fn launch_relu_mask(d_hidden: *mut f32, hidden: *const f32, total: i32);
    fn launch_col_sum(mat: *const f32, out_vec: *mut f32, rows: i32, cols: i32);
    fn launch_sgd(param: *mut f32, grad: *const f32, lr: f32, n: i32);
    fn launch_adam(param: *mut f32, grad: *const f32, m: *mut f32, v: *mut f32,
                   lr: f32, beta1: f32, beta2: f32, eps: f32, t: i32, n: i32);
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
fn cuda_memset_zero(ptr: *mut f32, count: usize) {
    unsafe { cudaMemset(ptr as *mut _, 0, count * 4); }
}

// ---------------------------------------------------------------------------
//  CUDA Adam optimizer state
// ---------------------------------------------------------------------------

struct CudaAdamState {
    m: *mut f32,
    v: *mut f32,
}

impl CudaAdamState {
    fn new(n: i32) -> Self {
        let m = cuda_malloc(n as usize * std::mem::size_of::<f32>());
        let v = cuda_malloc(n as usize * std::mem::size_of::<f32>());
        cuda_memset_zero(m, n as usize);
        cuda_memset_zero(v, n as usize);
        CudaAdamState { m, v }
    }
}

impl Drop for CudaAdamState {
    fn drop(&mut self) {
        cuda_free(self.m);
        cuda_free(self.v);
    }
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

    fn forward_batch(&self, d_inputs: *const f32, bs: i32,
                     col1_b: *mut f32, conv1_b: *mut f32, pool1_b: *mut f32,
                     col2_b: *mut f32, conv2_b: *mut f32, pool2_b: *mut f32,
                     flat: *mut f32, fc1_out: *mut f32, fc2_out: *mut f32, out: *mut f32) {
        unsafe {
            // Conv1: input [1, B*784] -> col [25, B*576] -> conv [6, B*576] -> pool [6, B*144]
            launch_im2col_batch(d_inputs, col1_b, 1, bs, IN_H as i32, IN_W as i32, C1_K, C1_K, 1, C1_OH, C1_OW);
            launch_matmul_nn(self.conv1_w, col1_b, conv1_b, C1_OUT, bs * C1_COL_COLS, C1_COL_ROWS);
            launch_add_bias(conv1_b, self.conv1_b, C1_OUT, bs * C1_COL_COLS);
            launch_relu_forward(conv1_b, C1_OUT * bs * C1_COL_COLS);
            launch_avg_pool_forward_batch(conv1_b, pool1_b, C1_OUT, bs, C1_OH, C1_OW, P1_SIZE, P1_H, P1_W);

            // Conv2: pool1 [6, B*144] -> col [150, B*64] -> conv [16, B*64] -> pool [16, B*16]
            launch_im2col_batch(pool1_b, col2_b, C2_IN, bs, P1_H, P1_W, C2_K, C2_K, 1, C2_OH, C2_OW);
            launch_matmul_nn(self.conv2_w, col2_b, conv2_b, C2_OUT, bs * C2_COL_COLS, C2_COL_ROWS);
            launch_add_bias(conv2_b, self.conv2_b, C2_OUT, bs * C2_COL_COLS);
            launch_relu_forward(conv2_b, C2_OUT * bs * C2_COL_COLS);
            launch_avg_pool_forward_batch(conv2_b, pool2_b, C2_OUT, bs, C2_OH, C2_OW, P2_SIZE, P2_H, P2_W);

            // Transpose [C2_OUT, B*P2_H*P2_W] -> [B, FLAT_SIZE]
            launch_transpose_cb_to_bc(pool2_b, flat, C2_OUT, bs, P2_H * P2_W);
        }

        // FC layers (batch-major [B, features])
        unsafe { launch_matmul_nn(flat, self.fc1_w, fc1_out, bs, FC1_OUT, FLAT_SIZE); }
        unsafe { launch_bias_relu(fc1_out, self.fc1_b, bs * FC1_OUT, FC1_OUT); }

        unsafe { launch_matmul_nn(fc1_out, self.fc2_w, fc2_out, bs, FC2_OUT, FC1_OUT); }
        unsafe { launch_bias_relu(fc2_out, self.fc2_b, bs * FC2_OUT, FC2_OUT); }

        unsafe { launch_matmul_nn(fc2_out, self.out_w, out, bs, NUM_CLASSES, FC2_OUT); }
        unsafe { launch_bias_softmax(out, self.out_b, bs, NUM_CLASSES); }
    }

    fn train(&mut self, inputs: &[f32], targets: &[i32], num_samples: usize,
             batch_size: usize, num_epochs: usize, learning_rate: f32,
             optimizer: &str, scheduler: &str) {
        let use_adam = optimizer == "adam";
        let adam_states: Vec<CudaAdamState> = if use_adam {
            vec![
                CudaAdamState::new(C1_OUT * C1_COL_ROWS),
                CudaAdamState::new(C1_OUT),
                CudaAdamState::new(C2_OUT * C2_COL_ROWS),
                CudaAdamState::new(C2_OUT),
                CudaAdamState::new(FLAT_SIZE * FC1_OUT),
                CudaAdamState::new(FC1_OUT),
                CudaAdamState::new(FC1_OUT * FC2_OUT),
                CudaAdamState::new(FC2_OUT),
                CudaAdamState::new(FC2_OUT * NUM_CLASSES),
                CudaAdamState::new(NUM_CLASSES),
            ]
        } else {
            vec![]
        };
        let mut step: i32 = 0;
        let warmup = if scheduler == "cosine" {
            (num_epochs as f32 * 0.05).max(1.0) as usize
        } else {
            0
        };
        let input_size = IN_C * IN_H * IN_W;
        let d_inputs = cuda_malloc(num_samples * input_size * 4);
        let d_targets = cuda_malloc(num_samples * 4) as *mut i32;
        cuda_memcpy_h2d(d_inputs, &inputs[..num_samples * input_size]);
        cuda_memcpy_h2d_i32(d_targets, &targets[..num_samples]);

        let bs_max = batch_size as i32;

        // Forward buffers (channel-major)
        let col1_b = cuda_malloc((C1_COL_ROWS * bs_max * C1_COL_COLS) as usize * 4);
        let conv1_b_buf = cuda_malloc((C1_OUT * bs_max * C1_COL_COLS) as usize * 4);
        let pool1_b = cuda_malloc((C1_OUT * bs_max * P1_H * P1_W) as usize * 4);
        let col2_b = cuda_malloc((C2_COL_ROWS * bs_max * C2_COL_COLS) as usize * 4);
        let conv2_b_buf = cuda_malloc((C2_OUT * bs_max * C2_COL_COLS) as usize * 4);
        let pool2_b = cuda_malloc((C2_OUT * bs_max * P2_H * P2_W) as usize * 4);
        let flat = cuda_malloc(batch_size * FLAT_SIZE as usize * 4);
        let fc1_out = cuda_malloc(batch_size * FC1_OUT as usize * 4);
        let fc2_out = cuda_malloc(batch_size * FC2_OUT as usize * 4);
        let out = cuda_malloc(batch_size * NUM_CLASSES as usize * 4);

        // Backward buffers
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

        // Backward conv buffers (channel-major)
        let d_pool2_b = cuda_malloc((C2_OUT * bs_max * P2_H * P2_W) as usize * 4);
        let d_conv2_b = cuda_malloc((C2_OUT * bs_max * C2_COL_COLS) as usize * 4);
        let d_col2_b = cuda_malloc((C2_COL_ROWS * bs_max * C2_COL_COLS) as usize * 4);
        let d_pool1_b = cuda_malloc((C2_IN * bs_max * P1_H * P1_W) as usize * 4);
        let d_conv1_b = cuda_malloc((C1_OUT * bs_max * C1_COL_COLS) as usize * 4);

        let d_losses = cuda_malloc(batch_size * 4);
        let d_loss_sum = cuda_malloc_managed(4);

        let num_batches = (num_samples + batch_size - 1) / batch_size;

        for epoch in 0..num_epochs {
            let lr = if scheduler == "cosine" {
                nn_common::cosine_lr(epoch, num_epochs, learning_rate, warmup, 1e-6)
            } else {
                learning_rate
            };
            unsafe { *d_loss_sum = 0.0f32; }

            for batch in 0..num_batches {
                let start = batch * batch_size;
                let bs = batch_size.min(num_samples - start) as i32;

                let x = unsafe { d_inputs.add(start * input_size) };
                let y = unsafe { d_targets.add(start) };

                self.forward_batch(x, bs, col1_b, conv1_b_buf, pool1_b,
                                   col2_b, conv2_b_buf, pool2_b,
                                   flat, fc1_out, fc2_out, out);

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

                // Conv backward (batched)
                unsafe {
                    // Transpose d_flat [B, FLAT_SIZE] -> d_pool2_b [C2_OUT, B*P2_H*P2_W]
                    launch_transpose_bc_to_cb(d_flat, d_pool2_b, bs, C2_OUT, P2_H * P2_W);

                    // Conv2 backward
                    cuda_memset_zero(d_conv2_b, (C2_OUT * bs * C2_COL_COLS) as usize);
                    launch_avg_pool_backward_batch(d_pool2_b, d_conv2_b, C2_OUT, bs, C2_OH, C2_OW, P2_SIZE, P2_H, P2_W);
                    launch_relu_mask(d_conv2_b, conv2_b_buf, C2_OUT * bs * C2_COL_COLS);

                    // Weight gradient: d_conv2 [C2_OUT, B*C2_COL_COLS] @ col2^T [B*C2_COL_COLS, C2_COL_ROWS]
                    launch_matmul_nt(d_conv2_b, col2_b, g_conv2_w, C2_OUT, C2_COL_ROWS, bs * C2_COL_COLS);
                    launch_bias_grad(d_conv2_b, g_conv2_b, C2_OUT, bs * C2_COL_COLS);

                    // Input gradient: W2^T [C2_COL_ROWS, C2_OUT] @ d_conv2 [C2_OUT, B*C2_COL_COLS]
                    launch_matmul_tn(self.conv2_w, d_conv2_b, d_col2_b, C2_COL_ROWS, bs * C2_COL_COLS, C2_OUT);

                    cuda_memset_zero(d_pool1_b, (C2_IN * bs * P1_H * P1_W) as usize);
                    launch_col2im_batch(d_col2_b, d_pool1_b, C2_IN, bs, P1_H, P1_W, C2_K, C2_K, 1, C2_OH, C2_OW);

                    // Conv1 backward
                    cuda_memset_zero(d_conv1_b, (C1_OUT * bs * C1_COL_COLS) as usize);
                    launch_avg_pool_backward_batch(d_pool1_b, d_conv1_b, C1_OUT, bs, C1_OH, C1_OW, P1_SIZE, P1_H, P1_W);
                    launch_relu_mask(d_conv1_b, conv1_b_buf, C1_OUT * bs * C1_COL_COLS);

                    launch_matmul_nt(d_conv1_b, col1_b, g_conv1_w, C1_OUT, C1_COL_ROWS, bs * C1_COL_COLS);
                    launch_bias_grad(d_conv1_b, g_conv1_b, C1_OUT, bs * C1_COL_COLS);
                }

                let lr_s = lr / bs as f32;
                if use_adam {
                    step += 1;
                    unsafe {
                        launch_adam(self.conv1_w, g_conv1_w, adam_states[0].m, adam_states[0].v, lr_s, 0.9, 0.999, 1e-8, step, C1_OUT * C1_COL_ROWS);
                        launch_adam(self.conv1_b, g_conv1_b, adam_states[1].m, adam_states[1].v, lr_s, 0.9, 0.999, 1e-8, step, C1_OUT);
                        launch_adam(self.conv2_w, g_conv2_w, adam_states[2].m, adam_states[2].v, lr_s, 0.9, 0.999, 1e-8, step, C2_OUT * C2_COL_ROWS);
                        launch_adam(self.conv2_b, g_conv2_b, adam_states[3].m, adam_states[3].v, lr_s, 0.9, 0.999, 1e-8, step, C2_OUT);
                        launch_adam(self.fc1_w, g_fc1_w, adam_states[4].m, adam_states[4].v, lr_s, 0.9, 0.999, 1e-8, step, FLAT_SIZE * FC1_OUT);
                        launch_adam(self.fc1_b, g_fc1_b, adam_states[5].m, adam_states[5].v, lr_s, 0.9, 0.999, 1e-8, step, FC1_OUT);
                        launch_adam(self.fc2_w, g_fc2_w, adam_states[6].m, adam_states[6].v, lr_s, 0.9, 0.999, 1e-8, step, FC1_OUT * FC2_OUT);
                        launch_adam(self.fc2_b, g_fc2_b, adam_states[7].m, adam_states[7].v, lr_s, 0.9, 0.999, 1e-8, step, FC2_OUT);
                        launch_adam(self.out_w, g_out_w, adam_states[8].m, adam_states[8].v, lr_s, 0.9, 0.999, 1e-8, step, FC2_OUT * NUM_CLASSES);
                        launch_adam(self.out_b, g_out_b, adam_states[9].m, adam_states[9].v, lr_s, 0.9, 0.999, 1e-8, step, NUM_CLASSES);
                    }
                } else {
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
            }

            cuda_sync();
            println!("  Epoch {:4}  loss: {:.4}", epoch, unsafe { *d_loss_sum } / num_samples as f32);
        }

        cuda_sync();
        drop(adam_states);
        for p in [d_inputs, d_targets as *mut f32,
                   col1_b, conv1_b_buf, pool1_b, col2_b, conv2_b_buf, pool2_b,
                   flat, fc1_out, fc2_out, out,
                   d_out, d_fc2, d_fc1, d_flat,
                   g_out_w, g_out_b, g_fc2_w, g_fc2_b, g_fc1_w, g_fc1_b,
                   g_conv2_w, g_conv2_b, g_conv1_w, g_conv1_b,
                   d_pool2_b, d_conv2_b, d_col2_b, d_pool1_b, d_conv1_b,
                   d_losses, d_loss_sum] {
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

        let col1_b = cuda_malloc((C1_COL_ROWS * eval_bs * C1_COL_COLS) as usize * 4);
        let conv1_b_buf = cuda_malloc((C1_OUT * eval_bs * C1_COL_COLS) as usize * 4);
        let pool1_b = cuda_malloc((C1_OUT * eval_bs * P1_H * P1_W) as usize * 4);
        let col2_b = cuda_malloc((C2_COL_ROWS * eval_bs * C2_COL_COLS) as usize * 4);
        let conv2_b_buf = cuda_malloc((C2_OUT * eval_bs * C2_COL_COLS) as usize * 4);
        let pool2_b = cuda_malloc((C2_OUT * eval_bs * P2_H * P2_W) as usize * 4);
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
                               col1_b, conv1_b_buf, pool1_b,
                               col2_b, conv2_b_buf, pool2_b,
                               flat, fc1_out, fc2_out, out);

            unsafe {
                launch_eval_metrics(out, d_targets.add(start), d_losses, d_correct, bs, NUM_CLASSES);
                launch_reduce_sum(d_losses, d_loss_sum, bs);
            }
        }

        cuda_sync();
        let loss = unsafe { *d_loss_sum } / num_samples as f32;
        let accuracy = unsafe { *d_correct } as f32 / num_samples as f32 * 100.0;

        for p in [d_inputs, d_targets as *mut f32,
                   col1_b, conv1_b_buf, pool1_b, col2_b, conv2_b_buf, pool2_b,
                   flat, fc1_out, fc2_out, out,
                   d_losses, d_loss_sum, d_correct as *mut f32] {
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
    let learning_rate = if args.learning_rate > 0.0 { args.learning_rate } else { DEFAULT_LR };
    println!("\nTraining LeNet-5 ({} epochs, batch_size={}, lr={:.4})...",
             args.epochs, args.batch_size, learning_rate);

    let t_start = Instant::now();
    cnn.train(&dataset.inputs[..train_size * dataset.input_size],
              &dataset.labels[..train_size], train_size, args.batch_size, args.epochs, learning_rate,
              &args.optimizer, &args.scheduler);
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
