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
//  MLP struct -- GPU pointers, NO cuBLAS, multi-layer
// ---------------------------------------------------------------------------

struct GpuMlp {
    layer_sizes: Vec<i32>,
    weights: Vec<*mut f32>,
    biases: Vec<*mut f32>,
}

impl GpuMlp {
    fn new(input_size: usize, hidden_size: usize, output_size: usize,
           num_hidden_layers: usize) -> Self {
        let mut layer_sizes: Vec<i32> = Vec::with_capacity(num_hidden_layers + 2);
        layer_sizes.push(input_size as i32);
        for _ in 0..num_hidden_layers {
            layer_sizes.push(hidden_size as i32);
        }
        layer_sizes.push(output_size as i32);

        let nl = layer_sizes.len() - 1;
        let mut weights = Vec::with_capacity(nl);
        let mut biases = Vec::with_capacity(nl);

        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        for i in 0..nl {
            let fan_in = layer_sizes[i] as usize;
            let fan_out = layer_sizes[i + 1] as usize;

            let w = cuda_malloc_managed(fan_in * fan_out * std::mem::size_of::<f32>());
            let b = cuda_malloc_managed(fan_out * std::mem::size_of::<f32>());
            cuda_memset_zero(b, fan_out);

            let scale = (2.0f32 / fan_in as f32).sqrt();
            unsafe { launch_init_weights(w, (fan_in * fan_out) as i32, scale, seed + i as u64); }

            weights.push(w);
            biases.push(b);
        }

        cuda_sync();

        GpuMlp { layer_sizes, weights, biases }
    }

    fn num_layers(&self) -> usize {
        self.layer_sizes.len() - 1
    }

    fn num_hidden(&self) -> usize {
        self.num_layers() - 1
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
        let nl = self.num_layers();
        let num_hidden = self.num_hidden();
        let inp = self.layer_sizes[0];
        let hid = if num_hidden > 0 { self.layer_sizes[1] } else { 0 };
        let out = *self.layer_sizes.last().unwrap();

        let use_adam = optimizer == "adam";
        let adam_w: Vec<Option<CudaAdamState>> = (0..nl).map(|i| {
            if use_adam {
                Some(CudaAdamState::new(self.layer_sizes[i] * self.layer_sizes[i + 1]))
            } else {
                None
            }
        }).collect();
        let adam_b: Vec<Option<CudaAdamState>> = (0..nl).map(|i| {
            if use_adam {
                Some(CudaAdamState::new(self.layer_sizes[i + 1]))
            } else {
                None
            }
        }).collect();
        let mut step: i32 = 0;

        let d_inputs = cuda_malloc(num_samples * inp as usize * std::mem::size_of::<f32>());
        let d_targets = cuda_malloc(num_samples * std::mem::size_of::<i32>()) as *mut i32;
        cuda_memcpy_h2d(d_inputs, &inputs[..num_samples * inp as usize]);
        cuda_memcpy_h2d_i32(d_targets, &targets[..num_samples]);

        let d_acts: Vec<*mut f32> = (0..num_hidden)
            .map(|_| cuda_malloc(batch_size * hid as usize * std::mem::size_of::<f32>()))
            .collect();

        let d_output = cuda_malloc(batch_size * out as usize * std::mem::size_of::<f32>());
        let d_d_output = cuda_malloc(batch_size * out as usize * std::mem::size_of::<f32>());

        let max_dim = if hid > out { hid } else { out };
        let d_grad_cur = cuda_malloc(batch_size * max_dim as usize * std::mem::size_of::<f32>());
        let d_grad_prev = cuda_malloc(batch_size * max_dim as usize * std::mem::size_of::<f32>());

        let d_grad_w: Vec<*mut f32> = (0..nl)
            .map(|i| cuda_malloc(self.layer_sizes[i] as usize * self.layer_sizes[i + 1] as usize * std::mem::size_of::<f32>()))
            .collect();
        let d_grad_b: Vec<*mut f32> = (0..nl)
            .map(|i| cuda_malloc(self.layer_sizes[i + 1] as usize * std::mem::size_of::<f32>()))
            .collect();

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

            unsafe { *d_loss_sum = 0.0f32; }

            for batch in 0..num_batches {
                let start = batch * batch_size;
                let bs = batch_size.min(num_samples - start);
                let bs_i = bs as i32;

                let x = unsafe { d_inputs.add(start * inp as usize) };
                let y = unsafe { d_targets.add(start) };

                // ---- Forward ----
                for i in 0..nl {
                    let in_dim = self.layer_sizes[i];
                    let out_dim = self.layer_sizes[i + 1];
                    let input_act: *const f32 = if i == 0 { x } else { d_acts[i - 1] };
                    let out_act: *mut f32 = if i < num_hidden { d_acts[i] } else { d_output };

                    unsafe {
                        launch_matmul_nn(input_act, self.weights[i], out_act, bs_i, out_dim, in_dim);
                    }

                    if i < num_hidden {
                        let n_elem = bs_i * out_dim;
                        unsafe { launch_bias_relu(out_act, self.biases[i], n_elem, out_dim); }
                    } else {
                        unsafe { launch_bias_softmax(out_act, self.biases[i], bs_i, out_dim); }
                    }
                }

                // ---- Loss ----
                unsafe {
                    launch_ce_grad(d_output, y, d_d_output, d_losses, bs_i, out);
                    launch_reduce_sum(d_losses, d_loss_sum, bs_i);
                }

                // ---- Backward ----
                let mut dcur_id: u8 = 0; // 0=d_d_output, 1=d_grad_cur, 2=d_grad_prev

                for layer in (0..nl).rev() {
                    let fan_in = self.layer_sizes[layer];
                    let fan_out = self.layer_sizes[layer + 1];
                    let input_act: *const f32 = if layer == 0 { x } else { d_acts[layer - 1] };

                    let dcur: *const f32 = match dcur_id {
                        0 => d_d_output, 1 => d_grad_cur, _ => d_grad_prev,
                    };

                    unsafe {
                        launch_matmul_tn(input_act, dcur, d_grad_w[layer], fan_in, fan_out, bs_i);
                        launch_col_sum(dcur, d_grad_b[layer], bs_i, fan_out);
                    }

                    if layer > 0 {
                        let next_id: u8 = match dcur_id { 0 => 1, 1 => 2, _ => 1 };
                        let dnext: *mut f32 = if next_id == 1 { d_grad_cur } else { d_grad_prev };

                        unsafe {
                            launch_matmul_nt(dcur, self.weights[layer], dnext, bs_i, fan_in, fan_out);
                            launch_relu_mask(dnext, d_acts[layer - 1], bs_i * fan_in);
                        }
                        dcur_id = next_id;
                    }
                }

                // ---- Parameter update ----
                let lr_s = lr / bs as f32;
                if use_adam {
                    step += 1;
                    for i in 0..nl {
                        let n_w = self.layer_sizes[i] * self.layer_sizes[i + 1];
                        let aw = adam_w[i].as_ref().unwrap();
                        let ab = adam_b[i].as_ref().unwrap();
                        unsafe {
                            launch_adam(self.weights[i], d_grad_w[i], aw.m, aw.v,
                                        lr, 0.9, 0.999, 1e-8, step, n_w);
                            launch_adam(self.biases[i], d_grad_b[i], ab.m, ab.v,
                                        lr, 0.9, 0.999, 1e-8, step, self.layer_sizes[i + 1]);
                        }
                    }
                } else {
                    for i in 0..nl {
                        let n_w = self.layer_sizes[i] * self.layer_sizes[i + 1];
                        unsafe {
                            launch_sgd(self.weights[i], d_grad_w[i], lr_s, n_w);
                            launch_sgd(self.biases[i], d_grad_b[i], lr_s, self.layer_sizes[i + 1]);
                        }
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

        cuda_free(d_inputs);
        cuda_free(d_targets as *mut f32);
        for p in &d_acts { cuda_free(*p); }
        cuda_free(d_output);
        cuda_free(d_d_output);
        cuda_free(d_grad_cur);
        cuda_free(d_grad_prev);
        for p in &d_grad_w { cuda_free(*p); }
        for p in &d_grad_b { cuda_free(*p); }
        cuda_free(d_losses);
        cuda_free(d_loss_sum);
    }

    fn evaluate(&self, inputs: &[f32], targets: &[i32], num_samples: usize) -> (f32, f32) {
        let nl = self.num_layers();
        let num_hidden = self.num_hidden();
        let inp = self.layer_sizes[0];
        let hid = if num_hidden > 0 { self.layer_sizes[1] } else { 0 };
        let out = *self.layer_sizes.last().unwrap();

        let d_inputs = cuda_malloc(num_samples * inp as usize * std::mem::size_of::<f32>());
        let d_targets = cuda_malloc(num_samples * std::mem::size_of::<i32>()) as *mut i32;
        cuda_memcpy_h2d(d_inputs, &inputs[..num_samples * inp as usize]);
        cuda_memcpy_h2d_i32(d_targets, &targets[..num_samples]);

        let d_buf0 = if num_hidden > 0 {
            cuda_malloc(num_samples * hid as usize * std::mem::size_of::<f32>())
        } else {
            std::ptr::null_mut()
        };
        let d_buf1 = if num_hidden > 0 {
            cuda_malloc(num_samples * hid as usize * std::mem::size_of::<f32>())
        } else {
            std::ptr::null_mut()
        };

        let d_output = cuda_malloc(num_samples * out as usize * std::mem::size_of::<f32>());

        let ns = num_samples as i32;

        for i in 0..nl {
            let in_dim = self.layer_sizes[i];
            let out_dim = self.layer_sizes[i + 1];

            let input_act: *const f32 = if i == 0 {
                d_inputs
            } else if (i - 1) % 2 == 0 {
                d_buf0
            } else {
                d_buf1
            };

            let out_act: *mut f32 = if i < num_hidden {
                if i % 2 == 0 { d_buf0 } else { d_buf1 }
            } else {
                d_output
            };

            unsafe {
                launch_matmul_nn(input_act, self.weights[i], out_act, ns, out_dim, in_dim);
            }

            if i < num_hidden {
                let n_elem = ns * out_dim;
                unsafe { launch_bias_relu(out_act, self.biases[i], n_elem, out_dim); }
            } else {
                unsafe { launch_bias_softmax(out_act, self.biases[i], ns, out_dim); }
            }
        }

        let d_losses = cuda_malloc(num_samples * std::mem::size_of::<f32>());
        let d_loss_sum = cuda_malloc_managed(std::mem::size_of::<f32>());
        let d_correct = cuda_malloc_managed(std::mem::size_of::<i32>()) as *mut i32;

        unsafe {
            *d_loss_sum = 0.0f32;
            *d_correct = 0i32;
        }

        unsafe {
            launch_eval_metrics(d_output, d_targets, d_losses, d_correct, ns, out);
            launch_reduce_sum(d_losses, d_loss_sum, ns);
        }

        cuda_sync();

        let loss = unsafe { *d_loss_sum } / num_samples as f32;
        let accuracy = unsafe { *d_correct } as f32 / num_samples as f32 * 100.0;

        cuda_free(d_inputs);
        cuda_free(d_targets as *mut f32);
        if !d_buf0.is_null() { cuda_free(d_buf0); }
        if !d_buf1.is_null() { cuda_free(d_buf1); }
        cuda_free(d_output);
        cuda_free(d_losses);
        cuda_free(d_loss_sum);
        cuda_free(d_correct as *mut f32);

        (loss, accuracy)
    }
}

impl Drop for GpuMlp {
    fn drop(&mut self) {
        for w in &self.weights { cuda_free(*w); }
        for b in &self.biases { cuda_free(*b); }
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

    let train_size = (dataset.num_samples as f32 * 0.8) as usize;
    let test_size = dataset.num_samples - train_size;

    println!("Train: {} samples, Test: {} samples", train_size, test_size);

    let mut mlp = GpuMlp::new(dataset.input_size, args.hidden_size, dataset.output_size,
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
