use nn_common::{load_dataset, parse_args, shuffle_data};
use std::time::Instant;

const DEFAULT_LR: f32 = 0.32;

// LeNet-5 dimensions
const IN_C: usize = 1;
const IN_H: i32 = 28;
const IN_W: i32 = 28;
const C1_OUT: i32 = 6;
const C1_IN: i32 = 1;
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

type CublasHandle = *mut std::ffi::c_void;

#[repr(C)]
#[allow(dead_code)]
enum CublasOperation { N = 0, T = 1 }

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

extern "C" {
    fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: i32) -> i32;
    fn cudaMemset(devPtr: *mut std::ffi::c_void, value: i32, count: usize) -> i32;
    fn cudaMemsetAsync(devPtr: *mut std::ffi::c_void, value: i32, count: usize, stream: *mut std::ffi::c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;

    fn cublasCreate_v2(handle: *mut CublasHandle) -> i32;
    fn cublasDestroy_v2(handle: CublasHandle) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasSgemm_v2(handle: CublasHandle, transa: CublasOperation, transb: CublasOperation,
                       m: i32, n: i32, k: i32, alpha: *const f32, A: *const f32, lda: i32,
                       B: *const f32, ldb: i32, beta: *const f32, C: *mut f32, ldc: i32) -> i32;

    // NCHW direct conv kernels
    fn launch_conv_fwd_bias_relu_nchw(input: *const f32, weight: *const f32, bias: *const f32,
                                       output: *mut f32, B: i32, C_in: i32, H: i32, W: i32,
                                       C_out: i32, K: i32, OH: i32, OW: i32);
    fn launch_avg_pool_fwd_nchw(input: *const f32, output: *mut f32,
                                  B: i32, C: i32, H: i32, W: i32, pool: i32, PH: i32, PW: i32);
    fn launch_avg_pool_bwd_nchw(grad_output: *const f32, grad_input: *mut f32,
                                  B: i32, C: i32, H: i32, W: i32, pool: i32, PH: i32, PW: i32);
    fn launch_conv_input_grad_nchw(d_output: *const f32, weight: *const f32, d_input: *mut f32,
                                    B: i32, C_out: i32, C_in: i32, H: i32, W: i32, K: i32, OH: i32, OW: i32);
    fn launch_conv_weight_grad_nchw(d_output: *const f32, input: *const f32, grad_w: *mut f32,
                                     num_weights: i32, B: i32, C_out: i32, C_in: i32, H: i32, W: i32,
                                     K: i32, OH: i32, OW: i32);
    fn launch_conv_bias_grad_nchw(d_output: *const f32, grad_bias: *mut f32,
                                    C_out: i32, B: i32, OH: i32, OW: i32);

    // Optimized backward pass kernels
    fn launch_im2col_nchw_batch(input: *const f32, col: *mut f32,
                                 B: i32, C_in: i32, H: i32, W: i32, K: i32, OH: i32, OW: i32);
    fn launch_nchw_to_cm_bias_grad(input: *const f32, output: *mut f32, bias_grad: *mut f32,
                                    B: i32, C: i32, H: i32, W: i32);
    fn launch_col2im_gather_nchw(col: *const f32, output: *mut f32,
                                  B: i32, C_in: i32, H: i32, W: i32, K: i32, OH: i32, OW: i32);

    // Shared elementwise kernels
    fn launch_bias_relu(hidden: *mut f32, bias: *const f32, total: i32, hid: i32);
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
//  cuDNN FFI
// ---------------------------------------------------------------------------

type CudnnHandle = *mut std::ffi::c_void;
type CudnnTensorDescriptor = *mut std::ffi::c_void;
type CudnnFilterDescriptor = *mut std::ffi::c_void;
type CudnnConvolutionDescriptor = *mut std::ffi::c_void;
type CudnnActivationDescriptor = *mut std::ffi::c_void;

// cuDNN enums (as i32)
const CUDNN_DATA_FLOAT: i32 = 0;
const CUDNN_TENSOR_NCHW: i32 = 0;
const CUDNN_CROSS_CORRELATION: i32 = 1;
const CUDNN_ACTIVATION_RELU: i32 = 1;
const CUDNN_PROPAGATE_NAN: i32 = 0;

#[repr(C)]
struct CudnnConvFwdAlgoPerf {
    algo: i32,
    status: i32,
    time: f32,
    memory: usize,
    determinism: i32,
    math_type: i32,
    reserved: [i32; 3],
}

#[repr(C)]
struct CudnnConvBwdDataAlgoPerf {
    algo: i32,
    status: i32,
    time: f32,
    memory: usize,
    determinism: i32,
    math_type: i32,
    reserved: [i32; 3],
}

#[repr(C)]
struct CudnnConvBwdFilterAlgoPerf {
    algo: i32,
    status: i32,
    time: f32,
    memory: usize,
    determinism: i32,
    math_type: i32,
    reserved: [i32; 3],
}

extern "C" {
    fn cudnnCreate(handle: *mut CudnnHandle) -> i32;
    fn cudnnDestroy(handle: CudnnHandle) -> i32;

    fn cudnnCreateTensorDescriptor(desc: *mut CudnnTensorDescriptor) -> i32;
    fn cudnnSetTensor4dDescriptor(desc: CudnnTensorDescriptor, format: i32, data_type: i32,
                                   n: i32, c: i32, h: i32, w: i32) -> i32;
    fn cudnnDestroyTensorDescriptor(desc: CudnnTensorDescriptor) -> i32;

    fn cudnnCreateFilterDescriptor(desc: *mut CudnnFilterDescriptor) -> i32;
    fn cudnnSetFilter4dDescriptor(desc: CudnnFilterDescriptor, data_type: i32, format: i32,
                                   k: i32, c: i32, h: i32, w: i32) -> i32;
    fn cudnnDestroyFilterDescriptor(desc: CudnnFilterDescriptor) -> i32;

    fn cudnnCreateConvolutionDescriptor(desc: *mut CudnnConvolutionDescriptor) -> i32;
    fn cudnnSetConvolution2dDescriptor(desc: CudnnConvolutionDescriptor,
                                        pad_h: i32, pad_w: i32, stride_h: i32, stride_w: i32,
                                        dilation_h: i32, dilation_w: i32,
                                        mode: i32, compute_type: i32) -> i32;
    fn cudnnDestroyConvolutionDescriptor(desc: CudnnConvolutionDescriptor) -> i32;
    fn cudnnSetConvolutionMathType(desc: CudnnConvolutionDescriptor, math_type: i32) -> i32;

    fn cudnnCreateActivationDescriptor(desc: *mut CudnnActivationDescriptor) -> i32;
    fn cudnnSetActivationDescriptor(desc: CudnnActivationDescriptor,
                                     mode: i32, nan_prop: i32, coef: f64) -> i32;
    fn cudnnDestroyActivationDescriptor(desc: CudnnActivationDescriptor) -> i32;

    fn cudnnGetConvolutionForwardAlgorithm_v7(
        handle: CudnnHandle, src_desc: CudnnTensorDescriptor,
        filter_desc: CudnnFilterDescriptor, conv_desc: CudnnConvolutionDescriptor,
        dest_desc: CudnnTensorDescriptor, requested_count: i32, returned_count: *mut i32,
        perf: *mut CudnnConvFwdAlgoPerf) -> i32;
    fn cudnnGetConvolutionForwardWorkspaceSize(
        handle: CudnnHandle, src_desc: CudnnTensorDescriptor,
        filter_desc: CudnnFilterDescriptor, conv_desc: CudnnConvolutionDescriptor,
        dest_desc: CudnnTensorDescriptor, algo: i32, size: *mut usize) -> i32;

    fn cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle: CudnnHandle, filter_desc: CudnnFilterDescriptor,
        diff_desc: CudnnTensorDescriptor, conv_desc: CudnnConvolutionDescriptor,
        grad_desc: CudnnTensorDescriptor, requested_count: i32, returned_count: *mut i32,
        perf: *mut CudnnConvBwdDataAlgoPerf) -> i32;
    fn cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle: CudnnHandle, filter_desc: CudnnFilterDescriptor,
        diff_desc: CudnnTensorDescriptor, conv_desc: CudnnConvolutionDescriptor,
        grad_desc: CudnnTensorDescriptor, algo: i32, size: *mut usize) -> i32;

    fn cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle: CudnnHandle, src_desc: CudnnTensorDescriptor,
        diff_desc: CudnnTensorDescriptor, conv_desc: CudnnConvolutionDescriptor,
        grad_desc: CudnnFilterDescriptor, requested_count: i32, returned_count: *mut i32,
        perf: *mut CudnnConvBwdFilterAlgoPerf) -> i32;
    fn cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle: CudnnHandle, src_desc: CudnnTensorDescriptor,
        diff_desc: CudnnTensorDescriptor, conv_desc: CudnnConvolutionDescriptor,
        grad_desc: CudnnFilterDescriptor, algo: i32, size: *mut usize) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn cudnnConvolutionBiasActivationForward(
        handle: CudnnHandle, alpha1: *const f32,
        x_desc: CudnnTensorDescriptor, x: *const f32,
        w_desc: CudnnFilterDescriptor, w: *const f32,
        conv_desc: CudnnConvolutionDescriptor, algo: i32,
        workspace: *mut f32, workspace_size: usize,
        alpha2: *const f32, z_desc: CudnnTensorDescriptor, z: *const f32,
        bias_desc: CudnnTensorDescriptor, bias: *const f32,
        act_desc: CudnnActivationDescriptor,
        y_desc: CudnnTensorDescriptor, y: *mut f32) -> i32;

    fn cudnnConvolutionBackwardFilter(
        handle: CudnnHandle, alpha: *const f32,
        x_desc: CudnnTensorDescriptor, x: *const f32,
        dy_desc: CudnnTensorDescriptor, dy: *const f32,
        conv_desc: CudnnConvolutionDescriptor, algo: i32,
        workspace: *mut f32, workspace_size: usize,
        beta: *const f32, dw_desc: CudnnFilterDescriptor, dw: *mut f32) -> i32;

    fn cudnnConvolutionBackwardData(
        handle: CudnnHandle, alpha: *const f32,
        w_desc: CudnnFilterDescriptor, w: *const f32,
        dy_desc: CudnnTensorDescriptor, dy: *const f32,
        conv_desc: CudnnConvolutionDescriptor, algo: i32,
        workspace: *mut f32, workspace_size: usize,
        beta: *const f32, dx_desc: CudnnTensorDescriptor, dx: *mut f32) -> i32;

    fn cudnnConvolutionBackwardBias(
        handle: CudnnHandle, alpha: *const f32,
        dy_desc: CudnnTensorDescriptor, dy: *const f32,
        beta: *const f32, db_desc: CudnnTensorDescriptor, db: *mut f32) -> i32;
}

// ---------------------------------------------------------------------------
//  cuDNN state
// ---------------------------------------------------------------------------

struct CudnnState {
    handle: CudnnHandle,
    desc_input1: CudnnTensorDescriptor,
    desc_conv1_out: CudnnTensorDescriptor,
    desc_conv1_bias: CudnnTensorDescriptor,
    desc_conv1_filter: CudnnFilterDescriptor,
    desc_conv1_conv: CudnnConvolutionDescriptor,
    desc_pool1_out: CudnnTensorDescriptor,
    desc_conv2_out: CudnnTensorDescriptor,
    desc_conv2_bias: CudnnTensorDescriptor,
    desc_conv2_filter: CudnnFilterDescriptor,
    desc_conv2_conv: CudnnConvolutionDescriptor,
    desc_relu: CudnnActivationDescriptor,
    algo_fwd1: i32, algo_fwd2: i32,
    algo_bwd_data1: i32, algo_bwd_data2: i32,
    algo_bwd_filt1: i32, algo_bwd_filt2: i32,
    ws_fwd1: usize, ws_fwd2: usize,
    ws_bwd_data1: usize, ws_bwd_data2: usize,
    ws_bwd_filt1: usize, ws_bwd_filt2: usize,
    workspace: *mut f32,
    configured_bs: i32,
}

fn cudnn_check(status: i32, msg: &str) {
    if status != 0 { panic!("cuDNN error {} at {}", status, msg); }
}

impl CudnnState {
    fn new() -> Self {
        let mut handle: CudnnHandle = std::ptr::null_mut();
        cudnn_check(unsafe { cudnnCreate(&mut handle) }, "cudnnCreate");

        let mut desc_input1: CudnnTensorDescriptor = std::ptr::null_mut();
        let mut desc_conv1_out: CudnnTensorDescriptor = std::ptr::null_mut();
        let mut desc_conv1_bias: CudnnTensorDescriptor = std::ptr::null_mut();
        let mut desc_conv1_filter: CudnnFilterDescriptor = std::ptr::null_mut();
        let mut desc_conv1_conv: CudnnConvolutionDescriptor = std::ptr::null_mut();
        let mut desc_pool1_out: CudnnTensorDescriptor = std::ptr::null_mut();
        let mut desc_conv2_out: CudnnTensorDescriptor = std::ptr::null_mut();
        let mut desc_conv2_bias: CudnnTensorDescriptor = std::ptr::null_mut();
        let mut desc_conv2_filter: CudnnFilterDescriptor = std::ptr::null_mut();
        let mut desc_conv2_conv: CudnnConvolutionDescriptor = std::ptr::null_mut();
        let mut desc_relu: CudnnActivationDescriptor = std::ptr::null_mut();

        unsafe {
            cudnn_check(cudnnCreateTensorDescriptor(&mut desc_input1), "createTensor input1");
            cudnn_check(cudnnCreateTensorDescriptor(&mut desc_conv1_out), "createTensor conv1_out");
            cudnn_check(cudnnCreateTensorDescriptor(&mut desc_conv1_bias), "createTensor conv1_bias");
            cudnn_check(cudnnCreateFilterDescriptor(&mut desc_conv1_filter), "createFilter conv1");
            cudnn_check(cudnnCreateConvolutionDescriptor(&mut desc_conv1_conv), "createConv conv1");
            cudnn_check(cudnnCreateTensorDescriptor(&mut desc_pool1_out), "createTensor pool1_out");
            cudnn_check(cudnnCreateTensorDescriptor(&mut desc_conv2_out), "createTensor conv2_out");
            cudnn_check(cudnnCreateTensorDescriptor(&mut desc_conv2_bias), "createTensor conv2_bias");
            cudnn_check(cudnnCreateFilterDescriptor(&mut desc_conv2_filter), "createFilter conv2");
            cudnn_check(cudnnCreateConvolutionDescriptor(&mut desc_conv2_conv), "createConv conv2");
            cudnn_check(cudnnCreateActivationDescriptor(&mut desc_relu), "createActivation");

            cudnn_check(cudnnSetActivationDescriptor(desc_relu, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0), "setActivation");

            cudnn_check(cudnnSetFilter4dDescriptor(desc_conv1_filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                C1_OUT, C1_IN, C1_K, C1_K), "setFilter conv1");
            cudnn_check(cudnnSetConvolution2dDescriptor(desc_conv1_conv,
                0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "setConv conv1");
            cudnn_check(cudnnSetConvolutionMathType(desc_conv1_conv, 2), "setMathType conv1"); // CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
            cudnn_check(cudnnSetTensor4dDescriptor(desc_conv1_bias, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                1, C1_OUT, 1, 1), "setBias conv1");

            cudnn_check(cudnnSetFilter4dDescriptor(desc_conv2_filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                C2_OUT, C2_IN, C2_K, C2_K), "setFilter conv2");
            cudnn_check(cudnnSetConvolution2dDescriptor(desc_conv2_conv,
                0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "setConv conv2");
            cudnn_check(cudnnSetConvolutionMathType(desc_conv2_conv, 2), "setMathType conv2");
            cudnn_check(cudnnSetTensor4dDescriptor(desc_conv2_bias, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                1, C2_OUT, 1, 1), "setBias conv2");
        }

        CudnnState {
            handle, desc_input1, desc_conv1_out, desc_conv1_bias, desc_conv1_filter, desc_conv1_conv,
            desc_pool1_out, desc_conv2_out, desc_conv2_bias, desc_conv2_filter, desc_conv2_conv, desc_relu,
            algo_fwd1: 0, algo_fwd2: 0, algo_bwd_data1: 0, algo_bwd_data2: 0,
            algo_bwd_filt1: 0, algo_bwd_filt2: 0,
            ws_fwd1: 0, ws_fwd2: 0, ws_bwd_data1: 0, ws_bwd_data2: 0,
            ws_bwd_filt1: 0, ws_bwd_filt2: 0,
            workspace: std::ptr::null_mut(),
            configured_bs: 0,
        }
    }

    fn setup(&mut self, bs: i32) {
        if bs == self.configured_bs { return; }
        self.configured_bs = bs;

        unsafe {
            cudnn_check(cudnnSetTensor4dDescriptor(self.desc_input1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                bs, C1_IN, IN_H, IN_W), "setTensor input1");
            cudnn_check(cudnnSetTensor4dDescriptor(self.desc_conv1_out, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                bs, C1_OUT, C1_OH, C1_OW), "setTensor conv1_out");
            cudnn_check(cudnnSetTensor4dDescriptor(self.desc_pool1_out, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                bs, C2_IN, P1_H, P1_W), "setTensor pool1_out");
            cudnn_check(cudnnSetTensor4dDescriptor(self.desc_conv2_out, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                bs, C2_OUT, C2_OH, C2_OW), "setTensor conv2_out");

            let mut returned: i32 = 0;
            let mut fwd_perf = std::mem::zeroed::<CudnnConvFwdAlgoPerf>();
            let mut bwd_data_perf = std::mem::zeroed::<CudnnConvBwdDataAlgoPerf>();
            let mut bwd_filt_perf = std::mem::zeroed::<CudnnConvBwdFilterAlgoPerf>();

            // Conv1 algorithms
            cudnn_check(cudnnGetConvolutionForwardAlgorithm_v7(self.handle,
                self.desc_input1, self.desc_conv1_filter, self.desc_conv1_conv, self.desc_conv1_out,
                1, &mut returned, &mut fwd_perf), "getFwdAlgo conv1");
            self.algo_fwd1 = fwd_perf.algo;
            cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(self.handle,
                self.desc_input1, self.desc_conv1_filter, self.desc_conv1_conv, self.desc_conv1_out,
                self.algo_fwd1, &mut self.ws_fwd1), "getFwdWs conv1");

            cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm_v7(self.handle,
                self.desc_conv1_filter, self.desc_conv1_out, self.desc_conv1_conv, self.desc_input1,
                1, &mut returned, &mut bwd_data_perf), "getBwdDataAlgo conv1");
            self.algo_bwd_data1 = bwd_data_perf.algo;
            cudnn_check(cudnnGetConvolutionBackwardDataWorkspaceSize(self.handle,
                self.desc_conv1_filter, self.desc_conv1_out, self.desc_conv1_conv, self.desc_input1,
                self.algo_bwd_data1, &mut self.ws_bwd_data1), "getBwdDataWs conv1");

            cudnn_check(cudnnGetConvolutionBackwardFilterAlgorithm_v7(self.handle,
                self.desc_input1, self.desc_conv1_out, self.desc_conv1_conv, self.desc_conv1_filter,
                1, &mut returned, &mut bwd_filt_perf), "getBwdFiltAlgo conv1");
            self.algo_bwd_filt1 = bwd_filt_perf.algo;
            cudnn_check(cudnnGetConvolutionBackwardFilterWorkspaceSize(self.handle,
                self.desc_input1, self.desc_conv1_out, self.desc_conv1_conv, self.desc_conv1_filter,
                self.algo_bwd_filt1, &mut self.ws_bwd_filt1), "getBwdFiltWs conv1");

            // Conv2 algorithms
            cudnn_check(cudnnGetConvolutionForwardAlgorithm_v7(self.handle,
                self.desc_pool1_out, self.desc_conv2_filter, self.desc_conv2_conv, self.desc_conv2_out,
                1, &mut returned, &mut fwd_perf), "getFwdAlgo conv2");
            self.algo_fwd2 = fwd_perf.algo;
            cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(self.handle,
                self.desc_pool1_out, self.desc_conv2_filter, self.desc_conv2_conv, self.desc_conv2_out,
                self.algo_fwd2, &mut self.ws_fwd2), "getFwdWs conv2");

            cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm_v7(self.handle,
                self.desc_conv2_filter, self.desc_conv2_out, self.desc_conv2_conv, self.desc_pool1_out,
                1, &mut returned, &mut bwd_data_perf), "getBwdDataAlgo conv2");
            self.algo_bwd_data2 = bwd_data_perf.algo;
            cudnn_check(cudnnGetConvolutionBackwardDataWorkspaceSize(self.handle,
                self.desc_conv2_filter, self.desc_conv2_out, self.desc_conv2_conv, self.desc_pool1_out,
                self.algo_bwd_data2, &mut self.ws_bwd_data2), "getBwdDataWs conv2");

            cudnn_check(cudnnGetConvolutionBackwardFilterAlgorithm_v7(self.handle,
                self.desc_pool1_out, self.desc_conv2_out, self.desc_conv2_conv, self.desc_conv2_filter,
                1, &mut returned, &mut bwd_filt_perf), "getBwdFiltAlgo conv2");
            self.algo_bwd_filt2 = bwd_filt_perf.algo;
            cudnn_check(cudnnGetConvolutionBackwardFilterWorkspaceSize(self.handle,
                self.desc_pool1_out, self.desc_conv2_out, self.desc_conv2_conv, self.desc_conv2_filter,
                self.algo_bwd_filt2, &mut self.ws_bwd_filt2), "getBwdFiltWs conv2");

            // Allocate max workspace
            let ws_max = [self.ws_fwd1, self.ws_fwd2, self.ws_bwd_data1, self.ws_bwd_data2,
                          self.ws_bwd_filt1, self.ws_bwd_filt2].iter().copied().max().unwrap_or(0);
            if !self.workspace.is_null() { cuda_free(self.workspace); }
            if ws_max > 0 { self.workspace = cuda_malloc(ws_max); }
        }
    }
}

impl Drop for CudnnState {
    fn drop(&mut self) {
        unsafe {
            if !self.workspace.is_null() { cuda_free(self.workspace); }
            cudnnDestroyTensorDescriptor(self.desc_input1);
            cudnnDestroyTensorDescriptor(self.desc_conv1_out);
            cudnnDestroyTensorDescriptor(self.desc_conv1_bias);
            cudnnDestroyFilterDescriptor(self.desc_conv1_filter);
            cudnnDestroyConvolutionDescriptor(self.desc_conv1_conv);
            cudnnDestroyTensorDescriptor(self.desc_pool1_out);
            cudnnDestroyTensorDescriptor(self.desc_conv2_out);
            cudnnDestroyTensorDescriptor(self.desc_conv2_bias);
            cudnnDestroyFilterDescriptor(self.desc_conv2_filter);
            cudnnDestroyConvolutionDescriptor(self.desc_conv2_conv);
            cudnnDestroyActivationDescriptor(self.desc_relu);
            cudnnDestroy(self.handle);
        }
    }
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

fn cuda_free(ptr: *mut f32) { unsafe { cudaFree(ptr as *mut std::ffi::c_void); } }
fn cuda_sync() { let err = unsafe { cudaDeviceSynchronize() }; if err != 0 { panic!("sync failed: {}", err); } }

fn cuda_memcpy_h2d(dst: *mut f32, src: &[f32]) {
    unsafe { cudaMemcpy(dst as *mut _, src.as_ptr() as *const _, src.len() * 4, CUDA_MEMCPY_HOST_TO_DEVICE); }
}
fn cuda_memcpy_h2d_i32(dst: *mut i32, src: &[i32]) {
    unsafe { cudaMemcpy(dst as *mut _, src.as_ptr() as *const _, src.len() * 4, CUDA_MEMCPY_HOST_TO_DEVICE); }
}
fn cuda_memcpy_d2h_f32(dst: &mut f32, src: *const f32) {
    unsafe { cudaMemcpy(dst as *mut f32 as *mut _, src as *const _, 4, CUDA_MEMCPY_DEVICE_TO_HOST); }
}
fn cuda_memcpy_d2h_i32(dst: &mut i32, src: *const i32) {
    unsafe { cudaMemcpy(dst as *mut i32 as *mut _, src as *const _, 4, CUDA_MEMCPY_DEVICE_TO_HOST); }
}
fn cuda_memset_zero(ptr: *mut f32, count: usize) {
    unsafe { cudaMemset(ptr as *mut _, 0, count * 4); }
}
fn cuda_memset_async_zero(ptr: *mut f32, bytes: usize) {
    unsafe { cudaMemsetAsync(ptr as *mut _, 0, bytes, std::ptr::null_mut()); }
}

// cuBLAS GEMM wrappers
fn sgemm_nn(h: CublasHandle, m: i32, n: i32, k: i32, a: *const f32, b: *const f32, c: *mut f32) {
    let alpha: f32 = 1.0; let beta: f32 = 0.0;
    unsafe { cublasSgemm_v2(h, CublasOperation::N, CublasOperation::N, n, m, k, &alpha, b, n, a, k, &beta, c, n); }
}
fn sgemm_tn(h: CublasHandle, m: i32, n: i32, k: i32, a: *const f32, b: *const f32, c: *mut f32) {
    let alpha: f32 = 1.0; let beta: f32 = 0.0;
    unsafe { cublasSgemm_v2(h, CublasOperation::N, CublasOperation::T, n, m, k, &alpha, b, n, a, m, &beta, c, n); }
}
fn sgemm_nt(h: CublasHandle, m: i32, n: i32, k: i32, a: *const f32, b: *const f32, c: *mut f32) {
    let alpha: f32 = 1.0; let beta: f32 = 0.0;
    unsafe { cublasSgemm_v2(h, CublasOperation::T, CublasOperation::N, n, m, k, &alpha, b, k, a, k, &beta, c, n); }
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
        let m = cuda_malloc(n as usize * 4);
        let v = cuda_malloc(n as usize * 4);
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
    cublas: CublasHandle,
    cudnn: CudnnState,
}

impl GpuCnn {
    fn new() -> Self {
        let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();

        // Device memory (not managed)
        let conv1_w = cuda_malloc((C1_OUT * C1_COL_ROWS) as usize * 4);
        let conv1_b = cuda_malloc(C1_OUT as usize * 4);
        let conv2_w = cuda_malloc((C2_OUT * C2_COL_ROWS) as usize * 4);
        let conv2_b = cuda_malloc(C2_OUT as usize * 4);
        let fc1_w = cuda_malloc((FLAT_SIZE * FC1_OUT) as usize * 4);
        let fc1_b = cuda_malloc(FC1_OUT as usize * 4);
        let fc2_w = cuda_malloc((FC1_OUT * FC2_OUT) as usize * 4);
        let fc2_b = cuda_malloc(FC2_OUT as usize * 4);
        let out_w = cuda_malloc((FC2_OUT * NUM_CLASSES) as usize * 4);
        let out_b = cuda_malloc(NUM_CLASSES as usize * 4);

        cuda_memset_zero(conv1_b, C1_OUT as usize);
        cuda_memset_zero(conv2_b, C2_OUT as usize);
        cuda_memset_zero(fc1_b, FC1_OUT as usize);
        cuda_memset_zero(fc2_b, FC2_OUT as usize);
        cuda_memset_zero(out_b, NUM_CLASSES as usize);

        // Xavier init on GPU via launch_init_weights
        unsafe {
            launch_init_weights(conv1_w, C1_OUT * C1_COL_ROWS, (2.0f32 / C1_COL_ROWS as f32).sqrt(), seed);
            launch_init_weights(conv2_w, C2_OUT * C2_COL_ROWS, (2.0f32 / C2_COL_ROWS as f32).sqrt(), seed + 1);
            launch_init_weights(fc1_w, FLAT_SIZE * FC1_OUT, (2.0f32 / FLAT_SIZE as f32).sqrt(), seed + 2);
            launch_init_weights(fc2_w, FC1_OUT * FC2_OUT, (2.0f32 / FC1_OUT as f32).sqrt(), seed + 3);
            launch_init_weights(out_w, FC2_OUT * NUM_CLASSES, (2.0f32 / FC2_OUT as f32).sqrt(), seed + 4);
        }
        cuda_sync();

        let mut cublas: CublasHandle = std::ptr::null_mut();
        let err = unsafe { cublasCreate_v2(&mut cublas) };
        if err != 0 { panic!("cublasCreate failed: {}", err); }

        GpuCnn { conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, out_w, out_b, cublas, cudnn: CudnnState::new() }
    }

    fn forward_batch(&mut self, d_inputs: *const f32, bs: i32,
                     conv1_out: *mut f32, pool1_out: *mut f32,
                     conv2_out: *mut f32, pool2_out: *mut f32,
                     fc1_out: *mut f32, fc2_out: *mut f32, out: *mut f32,
                     use_cudnn: bool) {
        unsafe {
            if use_cudnn {
                self.cudnn.setup(bs);
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;

                cudnn_check(cudnnConvolutionBiasActivationForward(self.cudnn.handle,
                    &alpha, self.cudnn.desc_input1, d_inputs,
                    self.cudnn.desc_conv1_filter, self.conv1_w,
                    self.cudnn.desc_conv1_conv, self.cudnn.algo_fwd1,
                    self.cudnn.workspace, self.cudnn.ws_fwd1,
                    &beta, self.cudnn.desc_conv1_out, conv1_out,
                    self.cudnn.desc_conv1_bias, self.conv1_b,
                    self.cudnn.desc_relu, self.cudnn.desc_conv1_out, conv1_out), "fwd conv1");
            } else {
                launch_conv_fwd_bias_relu_nchw(d_inputs, self.conv1_w, self.conv1_b, conv1_out,
                    bs, C1_IN, IN_H, IN_W, C1_OUT, C1_K, C1_OH, C1_OW);
            }

            launch_avg_pool_fwd_nchw(conv1_out, pool1_out, bs, C1_OUT, C1_OH, C1_OW, P1_SIZE, P1_H, P1_W);

            if use_cudnn {
                let alpha: f32 = 1.0;
                let beta: f32 = 0.0;

                cudnn_check(cudnnConvolutionBiasActivationForward(self.cudnn.handle,
                    &alpha, self.cudnn.desc_pool1_out, pool1_out,
                    self.cudnn.desc_conv2_filter, self.conv2_w,
                    self.cudnn.desc_conv2_conv, self.cudnn.algo_fwd2,
                    self.cudnn.workspace, self.cudnn.ws_fwd2,
                    &beta, self.cudnn.desc_conv2_out, conv2_out,
                    self.cudnn.desc_conv2_bias, self.conv2_b,
                    self.cudnn.desc_relu, self.cudnn.desc_conv2_out, conv2_out), "fwd conv2");
            } else {
                launch_conv_fwd_bias_relu_nchw(pool1_out, self.conv2_w, self.conv2_b, conv2_out,
                    bs, C2_IN, P1_H, P1_W, C2_OUT, C2_K, C2_OH, C2_OW);
            }

            launch_avg_pool_fwd_nchw(conv2_out, pool2_out, bs, C2_OUT, C2_OH, C2_OW, P2_SIZE, P2_H, P2_W);
        }

        // FC layers — pool2_out is already [B, FLAT_SIZE] (NCHW = batch-major)
        sgemm_nn(self.cublas, bs, FC1_OUT, FLAT_SIZE, pool2_out, self.fc1_w, fc1_out);
        unsafe { launch_bias_relu(fc1_out, self.fc1_b, bs * FC1_OUT, FC1_OUT); }

        sgemm_nn(self.cublas, bs, FC2_OUT, FC1_OUT, fc1_out, self.fc2_w, fc2_out);
        unsafe { launch_bias_relu(fc2_out, self.fc2_b, bs * FC2_OUT, FC2_OUT); }

        sgemm_nn(self.cublas, bs, NUM_CLASSES, FC2_OUT, fc2_out, self.out_w, out);
        unsafe { launch_bias_softmax(out, self.out_b, bs, NUM_CLASSES); }
    }

    fn train(&mut self, inputs: &[f32], targets: &[i32], num_samples: usize,
             batch_size: usize, num_epochs: usize, learning_rate: f32,
             optimizer: &str, scheduler: &str, use_cudnn: bool) {
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

        let input_size = IN_C * IN_H as usize * IN_W as usize;
        let d_inputs = cuda_malloc(num_samples * input_size * 4);
        let d_targets = cuda_malloc(num_samples * 4) as *mut i32;
        cuda_memcpy_h2d(d_inputs, &inputs[..num_samples * input_size]);
        cuda_memcpy_h2d_i32(d_targets, &targets[..num_samples]);

        let bs_max = batch_size as i32;

        // Forward NCHW buffers
        let conv1_out = cuda_malloc((C1_OUT * bs_max * C1_OH * C1_OW) as usize * 4);
        let pool1_out = cuda_malloc((C1_OUT * bs_max * P1_H * P1_W) as usize * 4);
        let conv2_out = cuda_malloc((C2_OUT * bs_max * C2_OH * C2_OW) as usize * 4);
        let pool2_out = cuda_malloc(batch_size * FLAT_SIZE as usize * 4);
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

        // Backward conv NCHW buffers
        let d_conv2 = cuda_malloc((C2_OUT * bs_max * C2_OH * C2_OW) as usize * 4);
        let d_pool1 = cuda_malloc((C2_IN * bs_max * P1_H * P1_W) as usize * 4);
        let d_conv1 = cuda_malloc((C1_OUT * bs_max * C1_OH * C1_OW) as usize * 4);

        // Loss tracking (device only)
        let d_losses = cuda_malloc(batch_size * 4);
        let d_loss_sum = cuda_malloc(4);

        let num_batches = (num_samples + batch_size - 1) / batch_size;
        let warmup = if scheduler == "cosine" {
            (num_epochs as f32 * 0.05).max(1.0) as usize
        } else {
            0
        };
        let print_interval = if num_epochs <= 50 { 1 } else { 50 };

        for epoch in 0..num_epochs {
            let lr = if scheduler == "cosine" {
                nn_common::cosine_lr(epoch, num_epochs, learning_rate, warmup, 1e-6)
            } else {
                learning_rate
            };

            cuda_memset_async_zero(d_loss_sum, 4);

            for batch in 0..num_batches {
                let start = batch * batch_size;
                let bs = batch_size.min(num_samples - start) as i32;

                let x = unsafe { d_inputs.add(start * input_size) };
                let y = unsafe { d_targets.add(start) };

                self.forward_batch(x, bs, conv1_out, pool1_out, conv2_out, pool2_out,
                                   fc1_out, fc2_out, out, use_cudnn);

                unsafe {
                    launch_ce_grad(out, y, d_out, d_losses, bs, NUM_CLASSES);
                    launch_reduce_sum(d_losses, d_loss_sum, bs);
                }

                // FC backward
                sgemm_tn(self.cublas, FC2_OUT, NUM_CLASSES, bs, fc2_out, d_out, g_out_w);
                unsafe { launch_col_sum(d_out, g_out_b, bs, NUM_CLASSES); }

                sgemm_nt(self.cublas, bs, FC2_OUT, NUM_CLASSES, d_out, self.out_w, d_fc2);
                unsafe { launch_relu_mask(d_fc2, fc2_out, bs * FC2_OUT); }

                sgemm_tn(self.cublas, FC1_OUT, FC2_OUT, bs, fc1_out, d_fc2, g_fc2_w);
                unsafe { launch_col_sum(d_fc2, g_fc2_b, bs, FC2_OUT); }

                sgemm_nt(self.cublas, bs, FC1_OUT, FC2_OUT, d_fc2, self.fc2_w, d_fc1);
                unsafe { launch_relu_mask(d_fc1, fc1_out, bs * FC1_OUT); }

                sgemm_tn(self.cublas, FLAT_SIZE, FC1_OUT, bs, pool2_out, d_fc1, g_fc1_w);
                unsafe { launch_col_sum(d_fc1, g_fc1_b, bs, FC1_OUT); }

                sgemm_nt(self.cublas, bs, FLAT_SIZE, FC1_OUT, d_fc1, self.fc1_w, d_flat);

                // Conv backward
                unsafe {
                    launch_avg_pool_bwd_nchw(d_flat, d_conv2, bs, C2_OUT, C2_OH, C2_OW, P2_SIZE, P2_H, P2_W);
                    launch_relu_mask(d_conv2, conv2_out, bs * C2_OUT * C2_COL_COLS);

                    if use_cudnn {
                        let alpha: f32 = 1.0;
                        let beta: f32 = 0.0;

                        cudnn_check(cudnnConvolutionBackwardFilter(self.cudnn.handle,
                            &alpha, self.cudnn.desc_pool1_out, pool1_out,
                            self.cudnn.desc_conv2_out, d_conv2,
                            self.cudnn.desc_conv2_conv, self.cudnn.algo_bwd_filt2,
                            self.cudnn.workspace, self.cudnn.ws_bwd_filt2,
                            &beta, self.cudnn.desc_conv2_filter, g_conv2_w), "bwd filt conv2");

                        cudnn_check(cudnnConvolutionBackwardBias(self.cudnn.handle,
                            &alpha, self.cudnn.desc_conv2_out, d_conv2,
                            &beta, self.cudnn.desc_conv2_bias, g_conv2_b), "bwd bias conv2");

                        cudnn_check(cudnnConvolutionBackwardData(self.cudnn.handle,
                            &alpha, self.cudnn.desc_conv2_filter, self.conv2_w,
                            self.cudnn.desc_conv2_out, d_conv2,
                            self.cudnn.desc_conv2_conv, self.cudnn.algo_bwd_data2,
                            self.cudnn.workspace, self.cudnn.ws_bwd_data2,
                            &beta, self.cudnn.desc_pool1_out, d_pool1), "bwd data conv2");
                    } else {
                        launch_conv_weight_grad_nchw(d_conv2, pool1_out, g_conv2_w,
                            C2_OUT * C2_COL_ROWS, bs, C2_OUT, C2_IN, P1_H, P1_W, C2_K, C2_OH, C2_OW);
                        launch_conv_bias_grad_nchw(d_conv2, g_conv2_b, C2_OUT, bs, C2_OH, C2_OW);
                        launch_conv_input_grad_nchw(d_conv2, self.conv2_w, d_pool1,
                            bs, C2_OUT, C2_IN, P1_H, P1_W, C2_K, C2_OH, C2_OW);
                    }

                    launch_avg_pool_bwd_nchw(d_pool1, d_conv1, bs, C1_OUT, C1_OH, C1_OW, P1_SIZE, P1_H, P1_W);
                    launch_relu_mask(d_conv1, conv1_out, bs * C1_OUT * C1_COL_COLS);

                    if use_cudnn {
                        let alpha: f32 = 1.0;
                        let beta: f32 = 0.0;

                        cudnn_check(cudnnConvolutionBackwardFilter(self.cudnn.handle,
                            &alpha, self.cudnn.desc_input1, x,
                            self.cudnn.desc_conv1_out, d_conv1,
                            self.cudnn.desc_conv1_conv, self.cudnn.algo_bwd_filt1,
                            self.cudnn.workspace, self.cudnn.ws_bwd_filt1,
                            &beta, self.cudnn.desc_conv1_filter, g_conv1_w), "bwd filt conv1");

                        cudnn_check(cudnnConvolutionBackwardBias(self.cudnn.handle,
                            &alpha, self.cudnn.desc_conv1_out, d_conv1,
                            &beta, self.cudnn.desc_conv1_bias, g_conv1_b), "bwd bias conv1");
                    } else {
                        launch_conv_weight_grad_nchw(d_conv1, x, g_conv1_w,
                            C1_OUT * C1_COL_ROWS, bs, C1_OUT, C1_IN, IN_H, IN_W, C1_K, C1_OH, C1_OW);
                        launch_conv_bias_grad_nchw(d_conv1, g_conv1_b, C1_OUT, bs, C1_OH, C1_OW);
                    }
                }

                let lr_s = lr / bs as f32;
                if use_adam {
                    step += 1;
                    unsafe {
                        launch_adam(self.conv1_w, g_conv1_w, adam_states[0].m, adam_states[0].v, lr, 0.9, 0.999, 1e-8, step, C1_OUT * C1_COL_ROWS);
                        launch_adam(self.conv1_b, g_conv1_b, adam_states[1].m, adam_states[1].v, lr, 0.9, 0.999, 1e-8, step, C1_OUT);
                        launch_adam(self.conv2_w, g_conv2_w, adam_states[2].m, adam_states[2].v, lr, 0.9, 0.999, 1e-8, step, C2_OUT * C2_COL_ROWS);
                        launch_adam(self.conv2_b, g_conv2_b, adam_states[3].m, adam_states[3].v, lr, 0.9, 0.999, 1e-8, step, C2_OUT);
                        launch_adam(self.fc1_w, g_fc1_w, adam_states[4].m, adam_states[4].v, lr, 0.9, 0.999, 1e-8, step, FLAT_SIZE * FC1_OUT);
                        launch_adam(self.fc1_b, g_fc1_b, adam_states[5].m, adam_states[5].v, lr, 0.9, 0.999, 1e-8, step, FC1_OUT);
                        launch_adam(self.fc2_w, g_fc2_w, adam_states[6].m, adam_states[6].v, lr, 0.9, 0.999, 1e-8, step, FC1_OUT * FC2_OUT);
                        launch_adam(self.fc2_b, g_fc2_b, adam_states[7].m, adam_states[7].v, lr, 0.9, 0.999, 1e-8, step, FC2_OUT);
                        launch_adam(self.out_w, g_out_w, adam_states[8].m, adam_states[8].v, lr, 0.9, 0.999, 1e-8, step, FC2_OUT * NUM_CLASSES);
                        launch_adam(self.out_b, g_out_b, adam_states[9].m, adam_states[9].v, lr, 0.9, 0.999, 1e-8, step, NUM_CLASSES);
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

            if epoch % print_interval == 0 || epoch == num_epochs - 1 {
                cuda_sync();
                let mut h_loss: f32 = 0.0;
                cuda_memcpy_d2h_f32(&mut h_loss, d_loss_sum);
                println!("  Epoch {:4}  loss: {:.4}", epoch, h_loss / num_samples as f32);
            }
        }

        cuda_sync();
        drop(adam_states);
        for p in [d_inputs, d_targets as *mut f32,
                   conv1_out, pool1_out, conv2_out, pool2_out,
                   fc1_out, fc2_out, out,
                   d_out, d_fc2, d_fc1, d_flat,
                   g_out_w, g_out_b, g_fc2_w, g_fc2_b, g_fc1_w, g_fc1_b,
                   g_conv2_w, g_conv2_b, g_conv1_w, g_conv1_b,
                   d_conv2, d_pool1, d_conv1,
                   d_losses, d_loss_sum] {
            cuda_free(p);
        }
    }

    fn evaluate(&mut self, inputs: &[f32], targets: &[i32], num_samples: usize, use_cudnn: bool) -> (f32, f32) {
        let input_size = IN_C * IN_H as usize * IN_W as usize;
        let eval_bs = 256i32;

        let d_inputs = cuda_malloc(num_samples * input_size * 4);
        let d_targets = cuda_malloc(num_samples * 4) as *mut i32;
        cuda_memcpy_h2d(d_inputs, &inputs[..num_samples * input_size]);
        cuda_memcpy_h2d_i32(d_targets, &targets[..num_samples]);

        let conv1_out = cuda_malloc((C1_OUT * eval_bs * C1_OH * C1_OW) as usize * 4);
        let pool1_out = cuda_malloc((C1_OUT * eval_bs * P1_H * P1_W) as usize * 4);
        let conv2_out = cuda_malloc((C2_OUT * eval_bs * C2_OH * C2_OW) as usize * 4);
        let pool2_out = cuda_malloc(eval_bs as usize * FLAT_SIZE as usize * 4);
        let fc1_out = cuda_malloc(eval_bs as usize * FC1_OUT as usize * 4);
        let fc2_out = cuda_malloc(eval_bs as usize * FC2_OUT as usize * 4);
        let out = cuda_malloc(eval_bs as usize * NUM_CLASSES as usize * 4);
        let d_losses = cuda_malloc(eval_bs as usize * 4);
        let d_loss_sum = cuda_malloc(4);
        let d_correct = cuda_malloc(4) as *mut i32;

        cuda_memset_zero(d_loss_sum, 1);
        cuda_memset_zero(d_correct as *mut f32, 1);

        let num_batches = (num_samples + eval_bs as usize - 1) / eval_bs as usize;
        for batch in 0..num_batches {
            let start = batch * eval_bs as usize;
            let bs = (eval_bs as usize).min(num_samples - start) as i32;

            self.forward_batch(unsafe { d_inputs.add(start * input_size) }, bs,
                               conv1_out, pool1_out, conv2_out, pool2_out,
                               fc1_out, fc2_out, out, use_cudnn);

            unsafe {
                launch_eval_metrics(out, d_targets.add(start), d_losses, d_correct, bs, NUM_CLASSES);
                launch_reduce_sum(d_losses, d_loss_sum, bs);
            }
        }

        cuda_sync();
        let mut h_loss: f32 = 0.0;
        let mut h_correct: i32 = 0;
        cuda_memcpy_d2h_f32(&mut h_loss, d_loss_sum);
        cuda_memcpy_d2h_i32(&mut h_correct, d_correct);
        let loss = h_loss / num_samples as f32;
        let accuracy = h_correct as f32 / num_samples as f32 * 100.0;

        for p in [d_inputs, d_targets as *mut f32,
                   conv1_out, pool1_out, conv2_out, pool2_out,
                   fc1_out, fc2_out, out,
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
        if !self.cublas.is_null() { unsafe { cublasDestroy_v2(self.cublas); } }
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
    let use_cudnn = args.backend == "cudnn";
    println!("\nTraining LeNet-5 ({} epochs, batch_size={}, lr={:.4})...",
             args.epochs, args.batch_size, learning_rate);

    let t_start = Instant::now();
    cnn.train(&dataset.inputs[..train_size * dataset.input_size],
              &dataset.labels[..train_size], train_size, args.batch_size, args.epochs, learning_rate,
              &args.optimizer, &args.scheduler, use_cudnn);
    let t_train = t_start.elapsed().as_secs_f64();

    let t_eval_start = Instant::now();
    let (loss, accuracy) = cnn.evaluate(
        &dataset.inputs[train_size * dataset.input_size..],
        &dataset.labels[train_size..], test_size, use_cudnn);
    let t_eval = t_eval_start.elapsed().as_secs_f64();

    println!("\n=== Results on {} ===", args.dataset);
    println!("Test Loss:     {:.4}", loss);
    println!("Test Accuracy: {:.2}%", accuracy);
    println!("Train time:    {:.3} s", t_train);
    println!("Eval time:     {:.3} s", t_eval);
    println!("Throughput:    {:.0} samples/s", train_size as f64 * args.epochs as f64 / t_train);
}
