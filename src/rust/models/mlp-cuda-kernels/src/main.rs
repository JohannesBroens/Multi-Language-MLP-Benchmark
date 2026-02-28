use nn_common::{load_dataset, normalize_features, parse_args, shuffle_data};
use std::time::Instant;

const DEFAULT_LR: f32 = 0.02;

// ---------------------------------------------------------------------------
//  CUDA FFI types and constants
// ---------------------------------------------------------------------------

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
#[allow(dead_code)]
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const CUDA_MANAGED_MEM_GLOBAL: u32 = 1; // cudaMemAttachGlobal

extern "C" {
    // CUDA runtime
    fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
    ) -> i32;
    fn cudaMemset(devPtr: *mut std::ffi::c_void, value: i32, count: usize) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaMallocManaged(devPtr: *mut *mut std::ffi::c_void, size: usize, flags: u32) -> i32;

    // Custom tiled matmul kernels (NO cuBLAS)
    fn launch_matmul_nn(A: *const f32, B: *const f32, C: *mut f32, M: i32, N: i32, K: i32);
    fn launch_matmul_tn(A: *const f32, B: *const f32, C: *mut f32, M: i32, N: i32, K: i32);
    fn launch_matmul_nt(A: *const f32, B: *const f32, C: *mut f32, M: i32, N: i32, K: i32);

    // Element-wise CUDA kernels
    fn launch_bias_relu(hidden: *mut f32, bias: *const f32, total: i32, hid: i32);
    fn launch_bias_softmax(output: *mut f32, bias: *const f32, bs: i32, out: i32);
    fn launch_ce_grad(
        output: *const f32,
        targets: *const i32,
        d_output: *mut f32,
        losses: *mut f32,
        bs: i32,
        out: i32,
    );
    fn launch_relu_mask(d_hidden: *mut f32, hidden: *const f32, total: i32);
    fn launch_col_sum(mat: *const f32, out_vec: *mut f32, rows: i32, cols: i32);
    fn launch_sgd(param: *mut f32, grad: *const f32, lr: f32, n: i32);
    fn launch_adam(param: *mut f32, grad: *const f32, m: *mut f32, v: *mut f32,
                   lr: f32, beta1: f32, beta2: f32, eps: f32, t: i32, n: i32);
    fn launch_reduce_sum(arr: *const f32, result: *mut f32, n: i32);
    fn launch_init_weights(weights: *mut f32, size: i32, scale: f32, seed: u64);
    fn launch_eval_metrics(
        output: *const f32,
        targets: *const i32,
        losses: *mut f32,
        correct: *mut i32,
        bs: i32,
        out: i32,
    );
}

// ---------------------------------------------------------------------------
//  CUDA helper functions
// ---------------------------------------------------------------------------

fn cuda_malloc(size: usize) -> *mut f32 {
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let err = unsafe { cudaMalloc(&mut ptr, size) };
    if err != 0 {
        panic!("cudaMalloc failed with error {}", err);
    }
    ptr as *mut f32
}

fn cuda_malloc_managed(size: usize) -> *mut f32 {
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let err = unsafe { cudaMallocManaged(&mut ptr, size, CUDA_MANAGED_MEM_GLOBAL) };
    if err != 0 {
        panic!("cudaMallocManaged failed with error {}", err);
    }
    ptr as *mut f32
}

fn cuda_free(ptr: *mut f32) {
    unsafe {
        cudaFree(ptr as *mut std::ffi::c_void);
    }
}

fn cuda_memcpy_h2d(dst: *mut f32, src: &[f32]) {
    let err = unsafe {
        cudaMemcpy(
            dst as *mut std::ffi::c_void,
            src.as_ptr() as *const std::ffi::c_void,
            src.len() * std::mem::size_of::<f32>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    };
    if err != 0 {
        panic!("cudaMemcpy H2D failed with error {}", err);
    }
}

fn cuda_memcpy_h2d_i32(dst: *mut i32, src: &[i32]) {
    let err = unsafe {
        cudaMemcpy(
            dst as *mut std::ffi::c_void,
            src.as_ptr() as *const std::ffi::c_void,
            src.len() * std::mem::size_of::<i32>(),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
    };
    if err != 0 {
        panic!("cudaMemcpy H2D i32 failed with error {}", err);
    }
}

fn cuda_memset_zero(ptr: *mut f32, count: usize) {
    unsafe {
        cudaMemset(
            ptr as *mut std::ffi::c_void,
            0,
            count * std::mem::size_of::<f32>(),
        );
    }
}

fn cuda_sync() {
    let err = unsafe { cudaDeviceSynchronize() };
    if err != 0 {
        panic!("cudaDeviceSynchronize failed with error {}", err);
    }
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
//  MLP struct -- GPU pointers, NO cuBLAS
// ---------------------------------------------------------------------------

struct GpuMlp {
    input_size: i32,
    hidden_size: i32,
    output_size: i32,
    // Managed memory (accessible from host + device)
    input_weights: *mut f32,  // [input_size  x hidden_size]
    output_weights: *mut f32, // [hidden_size x output_size]
    hidden_biases: *mut f32,  // [hidden_size]
    output_biases: *mut f32,  // [output_size]
}

impl GpuMlp {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let inp = input_size as i32;
        let hid = hidden_size as i32;
        let out = output_size as i32;

        // Managed memory for weights (accessible from both host and device)
        let input_weights =
            cuda_malloc_managed((input_size * hidden_size) * std::mem::size_of::<f32>());
        let output_weights =
            cuda_malloc_managed((hidden_size * output_size) * std::mem::size_of::<f32>());
        let hidden_biases = cuda_malloc_managed(hidden_size * std::mem::size_of::<f32>());
        let output_biases = cuda_malloc_managed(output_size * std::mem::size_of::<f32>());

        // Zero biases
        cuda_memset_zero(hidden_biases, hidden_size);
        cuda_memset_zero(output_biases, output_size);

        // Xavier init via GPU kernel
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let scale_ih = (2.0f32 / input_size as f32).sqrt();
        unsafe {
            launch_init_weights(input_weights, inp * hid, scale_ih, seed);
        }

        let scale_ho = (2.0f32 / hidden_size as f32).sqrt();
        unsafe {
            launch_init_weights(output_weights, hid * out, scale_ho, seed + 1);
        }

        cuda_sync();

        GpuMlp {
            input_size: inp,
            hidden_size: hid,
            output_size: out,
            input_weights,
            output_weights,
            hidden_biases,
            output_biases,
        }
    }

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
        let inp = self.input_size;
        let hid = self.hidden_size;
        let out = self.output_size;

        let use_adam = optimizer == "adam";
        let adam_w1 = if use_adam { Some(CudaAdamState::new(inp * hid)) } else { None };
        let adam_w2 = if use_adam { Some(CudaAdamState::new(hid * out)) } else { None };
        let adam_b1 = if use_adam { Some(CudaAdamState::new(hid)) } else { None };
        let adam_b2 = if use_adam { Some(CudaAdamState::new(out)) } else { None };
        let mut step: i32 = 0;

        // Copy training data to device
        let d_inputs = cuda_malloc(num_samples * inp as usize * std::mem::size_of::<f32>());
        let d_targets = cuda_malloc(num_samples * std::mem::size_of::<i32>()) as *mut i32;
        cuda_memcpy_h2d(d_inputs, &inputs[..num_samples * inp as usize]);
        cuda_memcpy_h2d_i32(d_targets, &targets[..num_samples]);

        // Workspace buffers on device
        let d_hidden = cuda_malloc(batch_size * hid as usize * std::mem::size_of::<f32>());
        let d_output = cuda_malloc(batch_size * out as usize * std::mem::size_of::<f32>());
        let d_d_output = cuda_malloc(batch_size * out as usize * std::mem::size_of::<f32>());
        let d_d_hidden = cuda_malloc(batch_size * hid as usize * std::mem::size_of::<f32>());
        let d_grad_w1 =
            cuda_malloc(inp as usize * hid as usize * std::mem::size_of::<f32>());
        let d_grad_w2 =
            cuda_malloc(hid as usize * out as usize * std::mem::size_of::<f32>());
        let d_grad_b1 = cuda_malloc(hid as usize * std::mem::size_of::<f32>());
        let d_grad_b2 = cuda_malloc(out as usize * std::mem::size_of::<f32>());
        let d_losses = cuda_malloc(batch_size * std::mem::size_of::<f32>());
        let d_loss_sum = cuda_malloc_managed(std::mem::size_of::<f32>());

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

            // Reset epoch loss
            unsafe {
                *d_loss_sum = 0.0f32;
            }

            for batch in 0..num_batches {
                let start = batch * batch_size;
                let bs = batch_size.min(num_samples - start);
                let bs_i = bs as i32;

                let x = unsafe { d_inputs.add(start * inp as usize) };
                let y = unsafe { (d_targets).add(start) };

                // ---- Forward: hidden = ReLU(X @ W1 + b1) ----
                unsafe {
                    launch_matmul_nn(x, self.input_weights, d_hidden, bs_i, hid, inp);
                }
                let n_hid = bs_i * hid;
                unsafe {
                    launch_bias_relu(d_hidden, self.hidden_biases, n_hid, hid);
                }

                // ---- Forward: output = softmax(hidden @ W2 + b2) ----
                unsafe {
                    launch_matmul_nn(d_hidden, self.output_weights, d_output, bs_i, out, hid);
                    launch_bias_softmax(d_output, self.output_biases, bs_i, out);
                }

                // ---- Loss + d_output gradient ----
                unsafe {
                    launch_ce_grad(d_output, y, d_d_output, d_losses, bs_i, out);
                }

                // Accumulate batch loss
                unsafe {
                    launch_reduce_sum(d_losses, d_loss_sum, bs_i);
                }

                // ---- Backward: matmul for gradients ----

                // grad_W2 = hidden^T @ d_output
                unsafe {
                    launch_matmul_tn(d_hidden, d_d_output, d_grad_w2, hid, out, bs_i);
                }

                // grad_b2 = colsum(d_output)
                unsafe {
                    launch_col_sum(d_d_output, d_grad_b2, bs_i, out);
                }

                // d_hidden = d_output @ W2^T
                unsafe {
                    launch_matmul_nt(d_d_output, self.output_weights, d_d_hidden, bs_i, hid, out);
                }

                // Apply ReLU mask
                unsafe {
                    launch_relu_mask(d_d_hidden, d_hidden, n_hid);
                }

                // grad_W1 = X^T @ d_hidden
                unsafe {
                    launch_matmul_tn(x, d_d_hidden, d_grad_w1, inp, hid, bs_i);
                }

                // grad_b1 = colsum(d_hidden)
                unsafe {
                    launch_col_sum(d_d_hidden, d_grad_b1, bs_i, hid);
                }

                // ---- Parameter update ----
                let lr_s = lr / bs as f32;
                let n_iw = inp * hid;
                let n_ow = hid * out;
                if let Some(ref a) = adam_w1 {
                    step += 1;
                    let aw2 = adam_w2.as_ref().unwrap();
                    let ab1 = adam_b1.as_ref().unwrap();
                    let ab2 = adam_b2.as_ref().unwrap();
                    unsafe {
                        launch_adam(self.input_weights, d_grad_w1, a.m, a.v, lr_s, 0.9, 0.999, 1e-8, step, n_iw);
                        launch_adam(self.output_weights, d_grad_w2, aw2.m, aw2.v, lr_s, 0.9, 0.999, 1e-8, step, n_ow);
                        launch_adam(self.hidden_biases, d_grad_b1, ab1.m, ab1.v, lr_s, 0.9, 0.999, 1e-8, step, hid);
                        launch_adam(self.output_biases, d_grad_b2, ab2.m, ab2.v, lr_s, 0.9, 0.999, 1e-8, step, out);
                    }
                } else {
                    unsafe {
                        launch_sgd(self.input_weights, d_grad_w1, lr_s, n_iw);
                        launch_sgd(self.output_weights, d_grad_w2, lr_s, n_ow);
                        launch_sgd(self.hidden_biases, d_grad_b1, lr_s, hid);
                        launch_sgd(self.output_biases, d_grad_b2, lr_s, out);
                    }
                }
            }

            if epoch % 100 == 0 {
                cuda_sync();
                let loss = unsafe { *d_loss_sum } / num_samples as f32;
                println!("  Epoch {:4}  loss: {:.4}", epoch, loss);
            }
        }

        cuda_sync();

        // Free workspace
        cuda_free(d_inputs);
        cuda_free(d_targets as *mut f32);
        cuda_free(d_hidden);
        cuda_free(d_output);
        cuda_free(d_d_output);
        cuda_free(d_d_hidden);
        cuda_free(d_grad_w1);
        cuda_free(d_grad_w2);
        cuda_free(d_grad_b1);
        cuda_free(d_grad_b2);
        cuda_free(d_losses);
        cuda_free(d_loss_sum);
    }

    fn evaluate(&self, inputs: &[f32], targets: &[i32], num_samples: usize) -> (f32, f32) {
        let inp = self.input_size;
        let hid = self.hidden_size;
        let out = self.output_size;

        // Copy test data to device
        let d_inputs = cuda_malloc(num_samples * inp as usize * std::mem::size_of::<f32>());
        let d_targets = cuda_malloc(num_samples * std::mem::size_of::<i32>()) as *mut i32;
        cuda_memcpy_h2d(d_inputs, &inputs[..num_samples * inp as usize]);
        cuda_memcpy_h2d_i32(d_targets, &targets[..num_samples]);

        // Workspace
        let d_hidden = cuda_malloc(num_samples * hid as usize * std::mem::size_of::<f32>());
        let d_output = cuda_malloc(num_samples * out as usize * std::mem::size_of::<f32>());
        let d_losses = cuda_malloc(num_samples * std::mem::size_of::<f32>());
        let d_loss_sum = cuda_malloc_managed(std::mem::size_of::<f32>());
        let d_correct = cuda_malloc_managed(std::mem::size_of::<i32>()) as *mut i32;

        unsafe {
            *d_loss_sum = 0.0f32;
            *d_correct = 0i32;
        }

        let ns = num_samples as i32;

        // Forward: hidden = ReLU(X @ W1 + b1)
        unsafe {
            launch_matmul_nn(d_inputs, self.input_weights, d_hidden, ns, hid, inp);
        }
        let n_hid = ns * hid;
        unsafe {
            launch_bias_relu(d_hidden, self.hidden_biases, n_hid, hid);
        }

        // Forward: output = softmax(hidden @ W2 + b2)
        unsafe {
            launch_matmul_nn(d_hidden, self.output_weights, d_output, ns, out, hid);
            launch_bias_softmax(d_output, self.output_biases, ns, out);
        }

        // Metrics
        unsafe {
            launch_eval_metrics(d_output, d_targets, d_losses, d_correct, ns, out);
            launch_reduce_sum(d_losses, d_loss_sum, ns);
        }

        cuda_sync();

        let loss = unsafe { *d_loss_sum } / num_samples as f32;
        let accuracy = unsafe { *d_correct } as f32 / num_samples as f32 * 100.0;

        // Free
        cuda_free(d_inputs);
        cuda_free(d_targets as *mut f32);
        cuda_free(d_hidden);
        cuda_free(d_output);
        cuda_free(d_losses);
        cuda_free(d_loss_sum);
        cuda_free(d_correct as *mut f32);

        (loss, accuracy)
    }
}

impl Drop for GpuMlp {
    fn drop(&mut self) {
        cuda_free(self.input_weights);
        cuda_free(self.output_weights);
        cuda_free(self.hidden_biases);
        cuda_free(self.output_biases);
    }
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------

fn main() {
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

    let mut mlp = GpuMlp::new(dataset.input_size, args.hidden_size, dataset.output_size);
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
