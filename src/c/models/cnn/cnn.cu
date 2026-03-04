/*
 * cnn.cu -- CUDA LeNet-5 CNN for MNIST digit classification.
 *
 * NCHW layout throughout.
 * Forward: cuDNN fused conv+bias+ReLU for convolutions, cuBLAS GEMM for FC.
 * Backward: cuDNN for conv weight/bias/data gradients, cuBLAS for FC.
 * Custom kernels for avg pool, softmax, cross-entropy, SGD/Adam.
 */

#include "cnn.h"
#include "nn_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#include <cudnn.h>

#define CUDNN_CHECK(call) do { \
    cudnnStatus_t stat = (call); \
    if (stat != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudnnGetErrorString(stat)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/* ------------------------------------------------------------------ */
/*  LeNet-5 layer dimensions                                          */
/* ------------------------------------------------------------------ */

#define IN_C  1
#define IN_H  28
#define IN_W  28

#define C1_OUT 6
#define C1_IN  1
#define C1_K   5
#define C1_OH  (IN_H - C1_K + 1)   /* 24 */
#define C1_OW  (IN_W - C1_K + 1)   /* 24 */
#define C1_COL_ROWS (C1_IN * C1_K * C1_K)  /* 25 */
#define C1_COL_COLS (C1_OH * C1_OW)         /* 576 */

#define P1_SIZE 2
#define P1_H   (C1_OH / P1_SIZE)   /* 12 */
#define P1_W   (C1_OW / P1_SIZE)   /* 12 */

#define C2_OUT 16
#define C2_IN  6
#define C2_K   5
#define C2_OH  (P1_H - C2_K + 1)   /* 8 */
#define C2_OW  (P1_W - C2_K + 1)   /* 8 */
#define C2_COL_ROWS (C2_IN * C2_K * C2_K)  /* 150 */
#define C2_COL_COLS (C2_OH * C2_OW)         /* 64 */

#define P2_SIZE 2
#define P2_H   (C2_OH / P2_SIZE)   /* 4 */
#define P2_W   (C2_OW / P2_SIZE)   /* 4 */

#define FLAT_SIZE (C2_OUT * P2_H * P2_W)  /* 256 */

#define FC1_OUT 120
#define FC2_OUT 84
#define NUM_CLASSES 10

/* ------------------------------------------------------------------ */
/*  cuDNN persistent state                                             */
/* ------------------------------------------------------------------ */

static cudnnHandle_t g_cudnn = NULL;

/* Descriptors for Conv1 and Conv2 — created once, batch dim updated per call */
static cudnnTensorDescriptor_t  desc_input1, desc_conv1_out, desc_conv1_bias;
static cudnnFilterDescriptor_t  desc_conv1_filter;
static cudnnConvolutionDescriptor_t desc_conv1_conv;
static cudnnActivationDescriptor_t  desc_relu;

static cudnnTensorDescriptor_t  desc_pool1_out, desc_conv2_out, desc_conv2_bias;
static cudnnFilterDescriptor_t  desc_conv2_filter;
static cudnnConvolutionDescriptor_t desc_conv2_conv;

/* Algorithm choices (found once per batch size) */
static cudnnConvolutionFwdAlgo_t    algo_fwd1, algo_fwd2;
static cudnnConvolutionBwdDataAlgo_t algo_bwd_data1, algo_bwd_data2;
static cudnnConvolutionBwdFilterAlgo_t algo_bwd_filt1, algo_bwd_filt2;
static size_t ws_fwd1, ws_fwd2, ws_bwd_data1, ws_bwd_data2, ws_bwd_filt1, ws_bwd_filt2;
static size_t ws_max = 0;
static float *d_workspace = NULL;
static int cudnn_bs = 0;  /* batch size that descriptors were configured for */

static void cudnn_setup(int bs) {
    if (!g_cudnn) {
        CUDNN_CHECK(cudnnCreate(&g_cudnn));

        /* Activation descriptor (ReLU, shared) */
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&desc_relu));
        CUDNN_CHECK(cudnnSetActivationDescriptor(desc_relu,
            CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

        /* Conv1 filter [C1_OUT, C1_IN, C1_K, C1_K] */
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc_conv1_filter));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(desc_conv1_filter,
            CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C1_OUT, C1_IN, C1_K, C1_K));

        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc_conv1_conv));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(desc_conv1_conv,
            0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CUDNN_CHECK(cudnnSetConvolutionMathType(desc_conv1_conv,
            CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_conv1_bias));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc_conv1_bias,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C1_OUT, 1, 1));

        /* Conv2 filter [C2_OUT, C2_IN, C2_K, C2_K] */
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc_conv2_filter));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(desc_conv2_filter,
            CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C2_OUT, C2_IN, C2_K, C2_K));

        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc_conv2_conv));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(desc_conv2_conv,
            0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CUDNN_CHECK(cudnnSetConvolutionMathType(desc_conv2_conv,
            CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_conv2_bias));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc_conv2_bias,
            CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C2_OUT, 1, 1));

        /* Tensor descriptors (batch-dependent — created here, set in if block below) */
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_input1));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_conv1_out));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_pool1_out));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc_conv2_out));
    }

    if (bs == cudnn_bs) return;
    cudnn_bs = bs;

    /* Set batch-dependent tensor descriptors */
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc_input1,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bs, C1_IN, IN_H, IN_W));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc_conv1_out,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bs, C1_OUT, C1_OH, C1_OW));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc_pool1_out,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bs, C2_IN, P1_H, P1_W));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc_conv2_out,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bs, C2_OUT, C2_OH, C2_OW));

    /* Find best algorithms */
    int returned;
    cudnnConvolutionFwdAlgoPerf_t fwd_perf;
    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf;
    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filt_perf;

    /* Conv1 algorithms */
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(g_cudnn,
        desc_input1, desc_conv1_filter, desc_conv1_conv, desc_conv1_out,
        1, &returned, &fwd_perf));
    algo_fwd1 = fwd_perf.algo;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(g_cudnn,
        desc_input1, desc_conv1_filter, desc_conv1_conv, desc_conv1_out,
        algo_fwd1, &ws_fwd1));

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(g_cudnn,
        desc_conv1_filter, desc_conv1_out, desc_conv1_conv, desc_input1,
        1, &returned, &bwd_data_perf));
    algo_bwd_data1 = bwd_data_perf.algo;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(g_cudnn,
        desc_conv1_filter, desc_conv1_out, desc_conv1_conv, desc_input1,
        algo_bwd_data1, &ws_bwd_data1));

    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(g_cudnn,
        desc_input1, desc_conv1_out, desc_conv1_conv, desc_conv1_filter,
        1, &returned, &bwd_filt_perf));
    algo_bwd_filt1 = bwd_filt_perf.algo;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(g_cudnn,
        desc_input1, desc_conv1_out, desc_conv1_conv, desc_conv1_filter,
        algo_bwd_filt1, &ws_bwd_filt1));

    /* Conv2 algorithms */
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(g_cudnn,
        desc_pool1_out, desc_conv2_filter, desc_conv2_conv, desc_conv2_out,
        1, &returned, &fwd_perf));
    algo_fwd2 = fwd_perf.algo;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(g_cudnn,
        desc_pool1_out, desc_conv2_filter, desc_conv2_conv, desc_conv2_out,
        algo_fwd2, &ws_fwd2));

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(g_cudnn,
        desc_conv2_filter, desc_conv2_out, desc_conv2_conv, desc_pool1_out,
        1, &returned, &bwd_data_perf));
    algo_bwd_data2 = bwd_data_perf.algo;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(g_cudnn,
        desc_conv2_filter, desc_conv2_out, desc_conv2_conv, desc_pool1_out,
        algo_bwd_data2, &ws_bwd_data2));

    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(g_cudnn,
        desc_pool1_out, desc_conv2_out, desc_conv2_conv, desc_conv2_filter,
        1, &returned, &bwd_filt_perf));
    algo_bwd_filt2 = bwd_filt_perf.algo;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(g_cudnn,
        desc_pool1_out, desc_conv2_out, desc_conv2_conv, desc_conv2_filter,
        algo_bwd_filt2, &ws_bwd_filt2));

    /* Allocate workspace */
    ws_max = ws_fwd1;
    if (ws_fwd2 > ws_max) ws_max = ws_fwd2;
    if (ws_bwd_data1 > ws_max) ws_max = ws_bwd_data1;
    if (ws_bwd_data2 > ws_max) ws_max = ws_bwd_data2;
    if (ws_bwd_filt1 > ws_max) ws_max = ws_bwd_filt1;
    if (ws_bwd_filt2 > ws_max) ws_max = ws_bwd_filt2;

    if (d_workspace) cudaFree(d_workspace);
    if (ws_max > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, ws_max));
    }
}

/* ------------------------------------------------------------------ */
/*  CUDA kernel helpers                                                */
/* ------------------------------------------------------------------ */

#define THREADS 256
#define BLOCKS(n) (((n) + THREADS - 1) / THREADS)

/* Warp-level reduction via shuffle — no shared memory, no __syncthreads */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

/* ------------------------------------------------------------------ */
/*  Per-sample kernels (cnn.h interface — kept for compatibility)      */
/* ------------------------------------------------------------------ */

__global__ void im2col_kernel(const float *input, float *col,
                               int C, int H, int W,
                               int kH, int kW, int stride,
                               int OH, int OW) {
    int total = C * kH * kW * OH * OW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int col_cols = OH * OW;
    int col_row = idx / col_cols;
    int col_col = idx % col_cols;

    int kw_idx = col_row % kW;
    int tmp = col_row / kW;
    int kh_idx = tmp % kH;
    int c = tmp / kH;

    int oh = col_col / OW;
    int ow = col_col % OW;

    int h = oh * stride + kh_idx;
    int w = ow * stride + kw_idx;

    col[idx] = input[c * H * W + h * W + w];
}

__global__ void col2im_kernel(const float *col, float *output,
                               int C, int H, int W,
                               int kH, int kW, int stride,
                               int OH, int OW) {
    int total = C * kH * kW * OH * OW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int col_cols = OH * OW;
    int col_row = idx / col_cols;
    int col_col = idx % col_cols;

    int kw_idx = col_row % kW;
    int tmp = col_row / kW;
    int kh_idx = tmp % kH;
    int c = tmp / kH;

    int oh = col_col / OW;
    int ow = col_col % OW;

    int h = oh * stride + kh_idx;
    int w = ow * stride + kw_idx;

    atomicAdd(&output[c * H * W + h * W + w], col[idx]);
}

__global__ void avg_pool_forward_kernel(const float *input, float *output,
                                         int C, int H, int W,
                                         int pool_size, int OH, int OW) {
    int total = C * OH * OW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int ow = idx % OW;
    int tmp = idx / OW;
    int oh = tmp % OH;
    int c = tmp / OH;

    float scale = 1.0f / (pool_size * pool_size);
    float sum = 0.0f;
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int h = oh * pool_size + ph;
            int w = ow * pool_size + pw;
            sum += input[c * H * W + h * W + w];
        }
    }
    output[idx] = sum * scale;
}

__global__ void avg_pool_backward_kernel(const float *grad_output, float *grad_input,
                                          int C, int H, int W,
                                          int pool_size, int OH, int OW) {
    int total = C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    int c = tmp / H;

    int oh = h / pool_size;
    int ow = w / pool_size;

    if (oh < OH && ow < OW) {
        float scale = 1.0f / (pool_size * pool_size);
        grad_input[idx] = grad_output[c * OH * OW + oh * OW + ow] * scale;
    }
}

/* ------------------------------------------------------------------ */
/*  NCHW batched kernels — direct convolution (no im2col)             */
/* ------------------------------------------------------------------ */

/*
 * Fused conv + bias + ReLU.  NCHW layout throughout.
 * input  [B, C_in, H, W]
 * weight [C_out, C_in * K * K]   (row-major)
 * bias   [C_out]
 * output [B, C_out, OH, OW]
 *
 * 2D grid: blockIdx.y = output channel, blockIdx.x = spatial tile (B*OH*OW).
 * Filter weights for one channel cached in shared memory.
 */
__global__ void conv_fwd_bias_relu_nchw(
        const float *input, const float *weight, const float *bias,
        float *output,
        int B, int C_in, int H, int W, int C_out, int K, int OH, int OW) {
    extern __shared__ float s_filter[];
    int CKK = C_in * K * K;
    int co = blockIdx.y;

    /* Cooperatively load filter for this output channel into shared mem */
    for (int i = threadIdx.x; i < CKK; i += blockDim.x)
        s_filter[i] = weight[co * CKK + i];
    __syncthreads();

    int spatial = B * OH * OW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= spatial) return;

    int ow_idx = idx % OW;
    int tmp = idx / OW;
    int oh_idx = tmp % OH;
    int b = tmp / OH;

    float sum = bias[co];
    for (int ci = 0; ci < C_in; ci++) {
        const float *in_bci = input + b * C_in * H * W + ci * H * W;
        const float *w_ci = s_filter + ci * K * K;
        for (int kh = 0; kh < K; kh++) {
            int ih = oh_idx + kh;
            for (int kw = 0; kw < K; kw++) {
                sum += in_bci[ih * W + (ow_idx + kw)] * w_ci[kh * K + kw];
            }
        }
    }
    output[b * C_out * OH * OW + co * OH * OW + oh_idx * OW + ow_idx] =
        (sum > 0.0f) ? sum : 0.0f;
}

/*
 * Batched im2col for NCHW input.  Extracts patches for all samples in a batch.
 * input [B, C_in, H, W]  -> col [C_in*K*K, B*OH*OW]
 *
 * Column layout: col[row][b*OH*OW + oh*OW + ow] where row = ci*K*K + kh*K + kw.
 * This layout lets us compute:
 *   weight[C_out, CKK] × col[CKK, B*OH*OW] = out[C_out, B*OH*OW]
 */
__global__ void im2col_nchw_batch_kernel(
        const float *input, float *col,
        int B, int C_in, int H, int W, int K, int OH, int OW) {
    int total = C_in * K * K * B * OH * OW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int col_cols = B * OH * OW;
    int col_row = idx / col_cols;
    int col_col = idx % col_cols;

    int kw = col_row % K;
    int tmp = col_row / K;
    int kh = tmp % K;
    int ci = tmp / K;

    int ow = col_col % OW;
    int tmp2 = col_col / OW;
    int oh = tmp2 % OH;
    int b  = tmp2 / OH;

    col[idx] = input[b * C_in * H * W + ci * H * W + (oh + kh) * W + (ow + kw)];
}

/*
 * Fused NCHW->CM transpose + bias gradient.
 * Transposes [B, C, H, W] -> [C, B*H*W] while accumulating per-channel sums.
 * bias_grad must be zeroed before launch (uses atomicAdd on global).
 * Shared memory: C floats for per-block partial sums.
 */
__global__ void nchw_to_cm_bias_grad_kernel(
        const float *input, float *output, float *bias_grad,
        int B, int C, int H, int W) {
    extern __shared__ float s_bias[];

    /* Zero shared bias accumulators */
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        s_bias[i] = 0.0f;
    __syncthreads();

    int total = B * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int w_idx = idx % W;
        int tmp = idx / W;
        int h_idx = tmp % H;
        int tmp2 = tmp / H;
        int c = tmp2 % C;
        int b = tmp2 / C;

        float val = input[idx];
        output[c * (B * H * W) + b * (H * W) + h_idx * W + w_idx] = val;
        atomicAdd(&s_bias[c], val);
    }
    __syncthreads();

    /* Write per-block partial sums to global */
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        atomicAdd(&bias_grad[i], s_bias[i]);
}

/*
 * Avg pool forward.  NCHW [B, C, H, W] -> [B, C, PH, PW].
 * One thread per output element.
 */
__global__ void avg_pool_fwd_nchw(
        const float *input, float *output,
        int B, int C, int H, int W, int pool, int PH, int PW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * PH * PW;
    if (idx >= total) return;

    int pw = idx % PW;
    int tmp = idx / PW;
    int ph = tmp % PH;
    int tmp2 = tmp / PH;
    int c = tmp2 % C;
    int b = tmp2 / C;

    float scale = 1.0f / (pool * pool);
    float sum = 0.0f;
    const float *in_bc = input + b * C * H * W + c * H * W;
    int h0 = ph * pool, w0 = pw * pool;
    for (int i = 0; i < pool; i++)
        for (int j = 0; j < pool; j++)
            sum += in_bc[(h0 + i) * W + w0 + j];
    output[idx] = sum * scale;
}

/*
 * Avg pool backward.  NCHW.
 * grad_output [B, C, PH, PW] -> grad_input [B, C, H, W].
 * One thread per grad_input element.
 */
__global__ void avg_pool_bwd_nchw(
        const float *grad_output, float *grad_input,
        int B, int C, int H, int W, int pool, int PH, int PW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    int tmp2 = tmp / H;
    int c = tmp2 % C;
    int b = tmp2 / C;

    int oh = h / pool;
    int ow = w / pool;
    if (oh < PH && ow < PW) {
        float scale = 1.0f / (pool * pool);
        grad_input[idx] = grad_output[b * C * PH * PW + c * PH * PW + oh * PW + ow] * scale;
    } else {
        grad_input[idx] = 0.0f;
    }
}

/*
 * Gather-based col2im for NCHW output.
 * Converts col [C_in*K*K, B*OH*OW] -> output [B, C_in, H, W].
 * Each thread gathers K*K values for one output element (no atomicAdd).
 */
__global__ void col2im_gather_nchw_kernel(
        const float *col, float *output,
        int B, int C_in, int H, int W, int K, int OH, int OW) {
    int total = B * C_in * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int iw = idx % W;
    int tmp = idx / W;
    int ih = tmp % H;
    int tmp2 = tmp / H;
    int ci = tmp2 % C_in;
    int b  = tmp2 / C_in;

    int N = B * OH * OW;
    float sum = 0.0f;
    for (int kh = 0; kh < K; kh++) {
        int oh = ih - kh;
        if (oh < 0 || oh >= OH) continue;
        for (int kw = 0; kw < K; kw++) {
            int ow_val = iw - kw;
            if (ow_val < 0 || ow_val >= OW) continue;
            int col_row = ci * K * K + kh * K + kw;
            int col_col = b * OH * OW + oh * OW + ow_val;
            sum += col[col_row * N + col_col];
        }
    }
    output[idx] = sum;
}

/*
 * Conv weight gradient via parallel reduction.  NCHW.
 * One block per weight element.  Threads reduce over B*OH*OW.
 *
 * d_output [B, C_out, OH, OW]
 * input    [B, C_in, H, W]
 * grad_w   [C_out, C_in * K * K]
 */
__global__ void conv_weight_grad_nchw(
        const float *d_output, const float *input, float *grad_w,
        int B, int C_out, int C_in, int H, int W, int K, int OH, int OW) {
    __shared__ float warp_sums[8];  /* max 256/32 = 8 warps */

    int w_idx = blockIdx.x;
    int CKK = C_in * K * K;
    int co = w_idx / CKK;
    int rem = w_idx % CKK;
    int ci = rem / (K * K);
    int kk = rem % (K * K);
    int kh = kk / K;
    int kw = kk % K;

    int spatial = B * OH * OW;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
        int b  = i / (OH * OW);
        int s  = i % (OH * OW);
        int oh = s / OW;
        int ow = s % OW;
        float d = d_output[b * C_out * OH * OW + co * OH * OW + oh * OW + ow];
        float x = input[b * C_in * H * W + ci * H * W + (oh + kh) * W + (ow + kw)];
        sum += d * x;
    }

    sum = warp_reduce_sum(sum);
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = blockDim.x >> 5;
        sum = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) grad_w[w_idx] = sum;
    }
}

/*
 * Conv bias gradient via parallel reduction.  NCHW.
 * One block per output channel.  Threads reduce over B*OH*OW.
 */
__global__ void conv_bias_grad_nchw(
        const float *d_output, float *grad_bias,
        int B, int C_out, int OH, int OW) {
    __shared__ float warp_sums[8];

    int co = blockIdx.x;
    int spatial = B * OH * OW;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
        int b  = i / (OH * OW);
        int s  = i % (OH * OW);
        int oh = s / OW;
        int ow = s % OW;
        sum += d_output[b * C_out * OH * OW + co * OH * OW + oh * OW + ow];
    }

    sum = warp_reduce_sum(sum);
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = blockDim.x >> 5;
        sum = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) grad_bias[co] = sum;
    }
}

/*
 * Conv input gradient.  NCHW.
 * d_output [B, C_out, OH, OW], weight [C_out, C_in*K*K]
 * -> d_input [B, C_in, H, W]
 * One thread per input element.
 */
__global__ void conv_input_grad_nchw(
        const float *d_output, const float *weight, float *d_input,
        int B, int C_out, int C_in, int H, int W, int K, int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C_in * H * W;
    if (idx >= total) return;

    int iw = idx % W;
    int tmp = idx / W;
    int ih = tmp % H;
    int tmp2 = tmp / H;
    int ci = tmp2 % C_in;
    int b  = tmp2 / C_in;

    int CKK = C_in * K * K;
    float sum = 0.0f;

    for (int co = 0; co < C_out; co++) {
        const float *d_bco = d_output + b * C_out * OH * OW + co * OH * OW;
        const float *w_coci = weight + co * CKK + ci * K * K;
        for (int kh = 0; kh < K; kh++) {
            int oh = ih - kh;
            if (oh < 0 || oh >= OH) continue;
            for (int kw = 0; kw < K; kw++) {
                int ow_val = iw - kw;
                if (ow_val < 0 || ow_val >= OW) continue;
                sum += d_bco[oh * OW + ow_val] * w_coci[kh * K + kw];
            }
        }
    }
    d_input[idx] = sum;
}

/*
 * Efficient bias gradient from channel-major [C, N] data.
 * Same algorithm as conv_bias_grad_nchw but reads contiguous memory.
 * One block per channel — contiguous reads make it fast despite fewer blocks.
 */
__global__ void bias_grad_cm_kernel(
        const float *data, float *result, int C, int N) {
    __shared__ float warp_sums[8];

    int co = blockIdx.x;
    const float *row = data + (size_t)co * N;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sum += row[i];

    sum = warp_reduce_sum(sum);
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = blockDim.x >> 5;
        sum = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) result[co] = sum;
    }
}

/* ------------------------------------------------------------------ */
/*  Shared utility kernels                                             */
/* ------------------------------------------------------------------ */

__global__ void eval_metrics_kernel(const float *output, const int *targets,
                                     float *losses, int *correct,
                                     int bs, int out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= bs) return;

    const float *o = output + row * out;
    int t = targets[row];
    losses[row] = -logf(o[t] + 1e-7f);

    int pred = 0;
    float mx = o[0];
    for (int j = 1; j < out; ++j) {
        if (o[j] > mx) { mx = o[j]; pred = j; }
    }
    if (pred == t) atomicAdd(correct, 1);
}

__global__ void cnn_reduce_sum_kernel(const float *arr, float *result, int n) {
    __shared__ float warp_sums[8];
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        sum += arr[i];

    sum = warp_reduce_sum(sum);
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = blockDim.x >> 5;
        sum = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) atomicAdd(result, sum);
    }
}

/* ------------------------------------------------------------------ */
/*  C interface for per-sample ops (cnn.h)                            */
/* ------------------------------------------------------------------ */

void im2col(const float *input, int C, int H, int W,
            int kH, int kW, int stride, float *col) {
    int OH = (H - kH) / stride + 1;
    int OW = (W - kW) / stride + 1;
    int total = C * kH * kW * OH * OW;
    im2col_kernel<<<BLOCKS(total), THREADS>>>(input, col, C, H, W,
                                               kH, kW, stride, OH, OW);
}

void col2im(const float *col, int C, int H, int W,
            int kH, int kW, int stride, float *output) {
    int OH = (H - kH) / stride + 1;
    int OW = (W - kW) / stride + 1;
    int total = C * kH * kW * OH * OW;
    col2im_kernel<<<BLOCKS(total), THREADS>>>(col, output, C, H, W,
                                               kH, kW, stride, OH, OW);
}

void avg_pool_forward(const float *input, float *output,
                      int C, int H, int W, int pool_size) {
    int OH = H / pool_size;
    int OW = W / pool_size;
    int total = C * OH * OW;
    avg_pool_forward_kernel<<<BLOCKS(total), THREADS>>>(input, output,
                                                         C, H, W,
                                                         pool_size, OH, OW);
}

void avg_pool_backward(const float *grad_output, float *grad_input,
                       int C, int H, int W, int pool_size) {
    int OH = H / pool_size;
    int OW = W / pool_size;
    int total = C * H * W;
    avg_pool_backward_kernel<<<BLOCKS(total), THREADS>>>(grad_output, grad_input,
                                                          C, H, W,
                                                          pool_size, OH, OW);
}

/* ------------------------------------------------------------------ */
/*  Initialize / Free                                                  */
/* ------------------------------------------------------------------ */

void cnn_initialize(CNN *cnn) {
    /* Allocate device memory (NOT managed) */
    CUDA_CHECK(cudaMalloc(&cnn->conv1_weights, C1_OUT * C1_COL_ROWS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cnn->conv1_biases,  C1_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cnn->conv2_weights, C2_OUT * C2_COL_ROWS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cnn->conv2_biases,  C2_OUT * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&cnn->fc1_weights, FLAT_SIZE * FC1_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cnn->fc1_biases,  FC1_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cnn->fc2_weights, FC1_OUT * FC2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cnn->fc2_biases,  FC2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cnn->out_weights, FC2_OUT * NUM_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cnn->out_biases,  NUM_CLASSES * sizeof(float)));

    cudaMemset(cnn->conv1_biases, 0, C1_OUT * sizeof(float));
    cudaMemset(cnn->conv2_biases, 0, C2_OUT * sizeof(float));
    cudaMemset(cnn->fc1_biases,   0, FC1_OUT * sizeof(float));
    cudaMemset(cnn->fc2_biases,   0, FC2_OUT * sizeof(float));
    cudaMemset(cnn->out_biases,   0, NUM_CLASSES * sizeof(float));

    /* Xavier init on CPU, then copy to device */
    int max_size = FLAT_SIZE * FC1_OUT;  /* largest weight tensor */
    float *h_tmp = (float *)malloc(max_size * sizeof(float));
    uint32_t rng = (uint32_t)time(NULL);

    nn_xavier_init(h_tmp, C1_OUT * C1_COL_ROWS, C1_COL_ROWS, &rng);
    CUDA_CHECK(cudaMemcpy(cnn->conv1_weights, h_tmp, C1_OUT * C1_COL_ROWS * sizeof(float), cudaMemcpyHostToDevice));

    nn_xavier_init(h_tmp, C2_OUT * C2_COL_ROWS, C2_COL_ROWS, &rng);
    CUDA_CHECK(cudaMemcpy(cnn->conv2_weights, h_tmp, C2_OUT * C2_COL_ROWS * sizeof(float), cudaMemcpyHostToDevice));

    nn_xavier_init(h_tmp, FLAT_SIZE * FC1_OUT, FLAT_SIZE, &rng);
    CUDA_CHECK(cudaMemcpy(cnn->fc1_weights, h_tmp, FLAT_SIZE * FC1_OUT * sizeof(float), cudaMemcpyHostToDevice));

    nn_xavier_init(h_tmp, FC1_OUT * FC2_OUT, FC1_OUT, &rng);
    CUDA_CHECK(cudaMemcpy(cnn->fc2_weights, h_tmp, FC1_OUT * FC2_OUT * sizeof(float), cudaMemcpyHostToDevice));

    nn_xavier_init(h_tmp, FC2_OUT * NUM_CLASSES, FC2_OUT, &rng);
    CUDA_CHECK(cudaMemcpy(cnn->out_weights, h_tmp, FC2_OUT * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice));

    free(h_tmp);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cnn_free(CNN *cnn) {
    cudaFree(cnn->conv1_weights); cudaFree(cnn->conv1_biases);
    cudaFree(cnn->conv2_weights); cudaFree(cnn->conv2_biases);
    cudaFree(cnn->fc1_weights);   cudaFree(cnn->fc1_biases);
    cudaFree(cnn->fc2_weights);   cudaFree(cnn->fc2_biases);
    cudaFree(cnn->out_weights);   cudaFree(cnn->out_biases);
}

/* ------------------------------------------------------------------ */
/*  Batched forward pass: direct conv (shared mem) + cuBLAS FC        */
/* ------------------------------------------------------------------ */

static void forward_batch(const CNN *cnn, const float *d_inputs, int bs,
                          float *conv1_out, float *pool1_out,
                          float *conv2_out, float *pool2_out,
                          float *fc1_out, float *fc2_out, float *out,
                          int use_cudnn) {
    int total;

    if (use_cudnn) {
        cudnn_setup(bs);
        const float alpha = 1.0f, beta = 0.0f;

        /* Conv1: fused conv + bias + ReLU via cuDNN */
        CUDNN_CHECK(cudnnConvolutionBiasActivationForward(g_cudnn,
            &alpha, desc_input1, d_inputs,
            desc_conv1_filter, cnn->conv1_weights,
            desc_conv1_conv, algo_fwd1, d_workspace, ws_fwd1,
            &beta, desc_conv1_out, conv1_out,
            desc_conv1_bias, cnn->conv1_biases,
            desc_relu, desc_conv1_out, conv1_out));
    } else {
        /* Conv1: custom fused conv + bias + ReLU kernel */
        int smem = C1_COL_ROWS * sizeof(float);
        int spatial = bs * C1_OH * C1_OW;
        dim3 grid1((spatial + THREADS - 1) / THREADS, C1_OUT);
        conv_fwd_bias_relu_nchw<<<grid1, THREADS, smem>>>(
            d_inputs, cnn->conv1_weights, cnn->conv1_biases, conv1_out,
            bs, C1_IN, IN_H, IN_W, C1_OUT, C1_K, C1_OH, C1_OW);
    }

    /* Pool1: [B,6,24,24] -> [B,6,12,12] */
    total = bs * C1_OUT * P1_H * P1_W;
    avg_pool_fwd_nchw<<<BLOCKS(total), THREADS>>>(
        conv1_out, pool1_out, bs, C1_OUT, C1_OH, C1_OW, P1_SIZE, P1_H, P1_W);

    if (use_cudnn) {
        const float alpha = 1.0f, beta = 0.0f;

        /* Conv2: fused conv + bias + ReLU via cuDNN */
        CUDNN_CHECK(cudnnConvolutionBiasActivationForward(g_cudnn,
            &alpha, desc_pool1_out, pool1_out,
            desc_conv2_filter, cnn->conv2_weights,
            desc_conv2_conv, algo_fwd2, d_workspace, ws_fwd2,
            &beta, desc_conv2_out, conv2_out,
            desc_conv2_bias, cnn->conv2_biases,
            desc_relu, desc_conv2_out, conv2_out));
    } else {
        /* Conv2: custom fused conv + bias + ReLU kernel */
        int smem = C2_COL_ROWS * sizeof(float);
        int spatial = bs * C2_OH * C2_OW;
        dim3 grid2((spatial + THREADS - 1) / THREADS, C2_OUT);
        conv_fwd_bias_relu_nchw<<<grid2, THREADS, smem>>>(
            pool1_out, cnn->conv2_weights, cnn->conv2_biases, conv2_out,
            bs, C2_IN, P1_H, P1_W, C2_OUT, C2_K, C2_OH, C2_OW);
    }

    /* Pool2: [B,16,8,8] -> [B,16,4,4] = [B,256] */
    total = bs * C2_OUT * P2_H * P2_W;
    avg_pool_fwd_nchw<<<BLOCKS(total), THREADS>>>(
        conv2_out, pool2_out, bs, C2_OUT, C2_OH, C2_OW, P2_SIZE, P2_H, P2_W);

    /* FC layers -- pool2_out is already [B, FLAT_SIZE] (NCHW = batch-major) */
    nn_sgemm_nn(bs, FC1_OUT, FLAT_SIZE, pool2_out, cnn->fc1_weights, fc1_out);
    nn_bias_relu(fc1_out, cnn->fc1_biases, bs, FC1_OUT);

    nn_sgemm_nn(bs, FC2_OUT, FC1_OUT, fc1_out, cnn->fc2_weights, fc2_out);
    nn_bias_relu(fc2_out, cnn->fc2_biases, bs, FC2_OUT);

    nn_sgemm_nn(bs, NUM_CLASSES, FC2_OUT, fc2_out, cnn->out_weights, out);
    nn_bias_softmax(out, cnn->out_biases, bs, NUM_CLASSES);
}

/* ------------------------------------------------------------------ */
/*  Training                                                           */
/* ------------------------------------------------------------------ */

void cnn_train(CNN *cnn, float *inputs, int *targets, int num_samples,
               int batch_size, int num_epochs, float learning_rate,
               OptimizerType optimizer, SchedulerType scheduler,
               int use_cudnn) {
    int input_size = IN_C * IN_H * IN_W;

    /* Copy training data to device */
    float *d_inputs;
    int   *d_targets;
    CUDA_CHECK(cudaMalloc(&d_inputs,  (size_t)num_samples * input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, (size_t)num_samples * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_inputs,  inputs,  (size_t)num_samples * input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets, (size_t)num_samples * sizeof(int), cudaMemcpyHostToDevice));

    /* Forward NCHW buffers */
    float *conv1_out, *pool1_out, *conv2_out, *pool2_out;
    CUDA_CHECK(cudaMalloc(&conv1_out, (size_t)batch_size * C1_OUT * C1_OH * C1_OW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pool1_out, (size_t)batch_size * C1_OUT * P1_H * P1_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&conv2_out, (size_t)batch_size * C2_OUT * C2_OH * C2_OW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pool2_out, (size_t)batch_size * FLAT_SIZE * sizeof(float)));

    /* FC layer buffers */
    float *fc1_out, *fc2_out, *out;
    CUDA_CHECK(cudaMalloc(&fc1_out, (size_t)batch_size * FC1_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fc2_out, (size_t)batch_size * FC2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out,     (size_t)batch_size * NUM_CLASSES * sizeof(float)));

    /* FC gradients */
    float *d_out, *d_fc2, *d_fc1, *d_flat;
    CUDA_CHECK(cudaMalloc(&d_out,  (size_t)batch_size * NUM_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc2,  (size_t)batch_size * FC2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc1,  (size_t)batch_size * FC1_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_flat, (size_t)batch_size * FLAT_SIZE * sizeof(float)));

    float *grad_out_w, *grad_out_b, *grad_fc2_w, *grad_fc2_b;
    float *grad_fc1_w, *grad_fc1_b;
    CUDA_CHECK(cudaMalloc(&grad_out_w, FC2_OUT * NUM_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_out_b, NUM_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_fc2_w, FC1_OUT * FC2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_fc2_b, FC2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_fc1_w, FLAT_SIZE * FC1_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_fc1_b, FC1_OUT * sizeof(float)));

    /* Conv gradients */
    float *grad_conv2_w, *grad_conv2_b, *grad_conv1_w, *grad_conv1_b;
    CUDA_CHECK(cudaMalloc(&grad_conv2_w, C2_OUT * C2_COL_ROWS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_conv2_b, C2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_conv1_w, C1_OUT * C1_COL_ROWS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_conv1_b, C1_OUT * sizeof(float)));

    /* Backward conv buffers (NCHW) */
    float *d_conv2, *d_pool1, *d_conv1;
    CUDA_CHECK(cudaMalloc(&d_conv2, (size_t)batch_size * C2_OUT * C2_OH * C2_OW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool1, (size_t)batch_size * C2_IN * P1_H * P1_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1, (size_t)batch_size * C1_OUT * C1_OH * C1_OW * sizeof(float)));

    /* Loss tracking (device only — no managed memory) */
    float *d_losses, *d_loss_sum;
    CUDA_CHECK(cudaMalloc(&d_losses,  (size_t)batch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss_sum, sizeof(float)));

    /* Adam optimizer state */
    AdamState adam_states[10] = {{0}};
    int step = 0;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    if (optimizer == OPT_ADAM) {
        int sizes[10] = {
            C1_OUT * C1_COL_ROWS, C1_OUT,
            C2_OUT * C2_COL_ROWS, C2_OUT,
            FLAT_SIZE * FC1_OUT,  FC1_OUT,
            FC1_OUT * FC2_OUT,    FC2_OUT,
            FC2_OUT * NUM_CLASSES, NUM_CLASSES
        };
        for (int i = 0; i < 10; i++)
            nn_adam_state_init(&adam_states[i], sizes[i]);
    }

    int num_batches = (num_samples + batch_size - 1) / batch_size;
    int warmup_epochs = (scheduler == SCHED_COSINE) ? (int)(num_epochs * 0.05f) : 0;
    if (warmup_epochs < 1 && scheduler == SCHED_COSINE) warmup_epochs = 1;

    float h_loss;
    int print_interval = (num_epochs <= 50) ? 1 : 50;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float lr = (scheduler == SCHED_COSINE)
            ? nn_cosine_lr(epoch, num_epochs, learning_rate, warmup_epochs, 1e-6f)
            : learning_rate;

        CUDA_CHECK(cudaMemsetAsync(d_loss_sum, 0, sizeof(float)));

        for (int batch = 0; batch < num_batches; batch++) {
            int start = batch * batch_size;
            int bs = batch_size;
            if (start + bs > num_samples) bs = num_samples - start;

            float *X = d_inputs + start * input_size;
            int   *Y = d_targets + start;
            int total;

            /* ---- Forward (direct conv + FC) ---- */
            forward_batch(cnn, X, bs,
                          conv1_out, pool1_out,
                          conv2_out, pool2_out,
                          fc1_out, fc2_out, out, use_cudnn);

            /* ---- Loss + output gradient ---- */
            nn_cross_entropy_grad(out, Y, d_out, d_losses, bs, NUM_CLASSES);
            cnn_reduce_sum_kernel<<<BLOCKS(bs), THREADS>>>(
                d_losses, d_loss_sum, bs);

            /* ---- FC backward ---- */
            nn_sgemm_tn(FC2_OUT, NUM_CLASSES, bs, fc2_out, d_out, grad_out_w);
            nn_col_sum(d_out, grad_out_b, bs, NUM_CLASSES);

            nn_sgemm_nt(bs, FC2_OUT, NUM_CLASSES, d_out, cnn->out_weights, d_fc2);
            nn_relu_backward(d_fc2, fc2_out, bs * FC2_OUT);

            nn_sgemm_tn(FC1_OUT, FC2_OUT, bs, fc1_out, d_fc2, grad_fc2_w);
            nn_col_sum(d_fc2, grad_fc2_b, bs, FC2_OUT);

            nn_sgemm_nt(bs, FC1_OUT, FC2_OUT, d_fc2, cnn->fc2_weights, d_fc1);
            nn_relu_backward(d_fc1, fc1_out, bs * FC1_OUT);

            nn_sgemm_tn(FLAT_SIZE, FC1_OUT, bs, pool2_out, d_fc1, grad_fc1_w);
            nn_col_sum(d_fc1, grad_fc1_b, bs, FC1_OUT);

            nn_sgemm_nt(bs, FLAT_SIZE, FC1_OUT, d_fc1, cnn->fc1_weights, d_flat);

            /* ---- Conv backward ---- */

            /* d_flat [B, 256] IS d_pool2 [B, 16, 4, 4] -- same memory layout */

            /* Pool2 backward -> d_conv2 [B, 16, 8, 8] */
            total = bs * C2_OUT * C2_OH * C2_OW;
            avg_pool_bwd_nchw<<<BLOCKS(total), THREADS>>>(
                d_flat, d_conv2, bs, C2_OUT, C2_OH, C2_OW, P2_SIZE, P2_H, P2_W);

            /* ReLU backward conv2 */
            nn_relu_backward(d_conv2, conv2_out, bs * C2_OUT * C2_COL_COLS);

            if (use_cudnn) {
                const float alpha = 1.0f, beta = 0.0f;

                CUDNN_CHECK(cudnnConvolutionBackwardFilter(g_cudnn,
                    &alpha, desc_pool1_out, pool1_out,
                    desc_conv2_out, d_conv2,
                    desc_conv2_conv, algo_bwd_filt2, d_workspace, ws_bwd_filt2,
                    &beta, desc_conv2_filter, grad_conv2_w));

                CUDNN_CHECK(cudnnConvolutionBackwardBias(g_cudnn,
                    &alpha, desc_conv2_out, d_conv2,
                    &beta, desc_conv2_bias, grad_conv2_b));

                CUDNN_CHECK(cudnnConvolutionBackwardData(g_cudnn,
                    &alpha, desc_conv2_filter, cnn->conv2_weights,
                    desc_conv2_out, d_conv2,
                    desc_conv2_conv, algo_bwd_data2, d_workspace, ws_bwd_data2,
                    &beta, desc_pool1_out, d_pool1));
            } else {
                conv_weight_grad_nchw<<<C2_OUT * C2_COL_ROWS, THREADS>>>(
                    d_conv2, pool1_out, grad_conv2_w,
                    bs, C2_OUT, C2_IN, P1_H, P1_W, C2_K, C2_OH, C2_OW);
                conv_bias_grad_nchw<<<C2_OUT, THREADS>>>(
                    d_conv2, grad_conv2_b, bs, C2_OUT, C2_OH, C2_OW);
                total = bs * C2_IN * P1_H * P1_W;
                conv_input_grad_nchw<<<BLOCKS(total), THREADS>>>(
                    d_conv2, cnn->conv2_weights, d_pool1,
                    bs, C2_OUT, C2_IN, P1_H, P1_W, C2_K, C2_OH, C2_OW);
            }

            /* Pool1 backward -> d_conv1 [B, 6, 24, 24] */
            total = bs * C1_OUT * C1_OH * C1_OW;
            avg_pool_bwd_nchw<<<BLOCKS(total), THREADS>>>(
                d_pool1, d_conv1, bs, C1_OUT, C1_OH, C1_OW, P1_SIZE, P1_H, P1_W);

            /* ReLU backward conv1 */
            nn_relu_backward(d_conv1, conv1_out, bs * C1_OUT * C1_COL_COLS);

            if (use_cudnn) {
                const float alpha = 1.0f, beta = 0.0f;

                CUDNN_CHECK(cudnnConvolutionBackwardFilter(g_cudnn,
                    &alpha, desc_input1, X,
                    desc_conv1_out, d_conv1,
                    desc_conv1_conv, algo_bwd_filt1, d_workspace, ws_bwd_filt1,
                    &beta, desc_conv1_filter, grad_conv1_w));

                CUDNN_CHECK(cudnnConvolutionBackwardBias(g_cudnn,
                    &alpha, desc_conv1_out, d_conv1,
                    &beta, desc_conv1_bias, grad_conv1_b));
            } else {
                conv_weight_grad_nchw<<<C1_OUT * C1_COL_ROWS, THREADS>>>(
                    d_conv1, X, grad_conv1_w,
                    bs, C1_OUT, C1_IN, IN_H, IN_W, C1_K, C1_OH, C1_OW);
                conv_bias_grad_nchw<<<C1_OUT, THREADS>>>(
                    d_conv1, grad_conv1_b, bs, C1_OUT, C1_OH, C1_OW);
            }

            /* ---- Parameter update ---- */
            float lr_s = lr / (float)bs;
            step++;

            if (optimizer == OPT_ADAM) {
                /* Adam's m/sqrt(v) ratio is scale-invariant, so use raw lr
                   (gradients are sums, but the ratio cancels the batch factor) */
                nn_adam_update(cnn->conv1_weights, grad_conv1_w, &adam_states[0], lr, beta1, beta2, eps, step);
                nn_adam_update(cnn->conv1_biases, grad_conv1_b, &adam_states[1], lr, beta1, beta2, eps, step);
                nn_adam_update(cnn->conv2_weights, grad_conv2_w, &adam_states[2], lr, beta1, beta2, eps, step);
                nn_adam_update(cnn->conv2_biases, grad_conv2_b, &adam_states[3], lr, beta1, beta2, eps, step);
                nn_adam_update(cnn->fc1_weights, grad_fc1_w, &adam_states[4], lr, beta1, beta2, eps, step);
                nn_adam_update(cnn->fc1_biases, grad_fc1_b, &adam_states[5], lr, beta1, beta2, eps, step);
                nn_adam_update(cnn->fc2_weights, grad_fc2_w, &adam_states[6], lr, beta1, beta2, eps, step);
                nn_adam_update(cnn->fc2_biases, grad_fc2_b, &adam_states[7], lr, beta1, beta2, eps, step);
                nn_adam_update(cnn->out_weights, grad_out_w, &adam_states[8], lr, beta1, beta2, eps, step);
                nn_adam_update(cnn->out_biases, grad_out_b, &adam_states[9], lr, beta1, beta2, eps, step);
            } else {
                nn_sgd_update(cnn->out_weights, grad_out_w, lr_s, FC2_OUT * NUM_CLASSES);
                nn_sgd_update(cnn->out_biases, grad_out_b, lr_s, NUM_CLASSES);
                nn_sgd_update(cnn->fc2_weights, grad_fc2_w, lr_s, FC1_OUT * FC2_OUT);
                nn_sgd_update(cnn->fc2_biases, grad_fc2_b, lr_s, FC2_OUT);
                nn_sgd_update(cnn->fc1_weights, grad_fc1_w, lr_s, FLAT_SIZE * FC1_OUT);
                nn_sgd_update(cnn->fc1_biases, grad_fc1_b, lr_s, FC1_OUT);
                nn_sgd_update(cnn->conv2_weights, grad_conv2_w, lr_s, C2_OUT * C2_COL_ROWS);
                nn_sgd_update(cnn->conv2_biases, grad_conv2_b, lr_s, C2_OUT);
                nn_sgd_update(cnn->conv1_weights, grad_conv1_w, lr_s, C1_OUT * C1_COL_ROWS);
                nn_sgd_update(cnn->conv1_biases, grad_conv1_b, lr_s, C1_OUT);
            }
        }

        if (epoch % print_interval == 0 || epoch == num_epochs - 1) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_loss, d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost));
            printf("  Epoch %4d  loss: %.4f\n", epoch, h_loss / num_samples);
            fflush(stdout);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (optimizer == OPT_ADAM) {
        for (int i = 0; i < 10; i++)
            nn_adam_state_free(&adam_states[i]);
    }

    /* Free workspace */
    cudaFree(d_inputs);    cudaFree(d_targets);
    cudaFree(conv1_out);   cudaFree(pool1_out);
    cudaFree(conv2_out);   cudaFree(pool2_out);
    cudaFree(fc1_out);     cudaFree(fc2_out);    cudaFree(out);
    cudaFree(d_out);       cudaFree(d_fc2);      cudaFree(d_fc1);      cudaFree(d_flat);
    cudaFree(grad_out_w);  cudaFree(grad_out_b);
    cudaFree(grad_fc2_w);  cudaFree(grad_fc2_b);
    cudaFree(grad_fc1_w);  cudaFree(grad_fc1_b);
    cudaFree(grad_conv2_w); cudaFree(grad_conv2_b);
    cudaFree(grad_conv1_w); cudaFree(grad_conv1_b);
    cudaFree(d_conv2);     cudaFree(d_pool1);    cudaFree(d_conv1);
    cudaFree(d_losses);    cudaFree(d_loss_sum);
}

/* ------------------------------------------------------------------ */
/*  Evaluation                                                         */
/* ------------------------------------------------------------------ */

void cnn_evaluate(CNN *cnn, float *inputs, int *targets, int num_samples,
                  float *loss, float *accuracy, int use_cudnn) {
    int input_size = IN_C * IN_H * IN_W;
    int eval_bs = 256;

    /* Copy data to device */
    float *d_inputs;
    int   *d_targets;
    CUDA_CHECK(cudaMalloc(&d_inputs,  (size_t)num_samples * input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, (size_t)num_samples * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_inputs,  inputs,  (size_t)num_samples * input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets, (size_t)num_samples * sizeof(int), cudaMemcpyHostToDevice));

    /* NCHW forward buffers */
    float *conv1_out, *pool1_out, *conv2_out, *pool2_out;
    CUDA_CHECK(cudaMalloc(&conv1_out, (size_t)eval_bs * C1_OUT * C1_OH * C1_OW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pool1_out, (size_t)eval_bs * C1_OUT * P1_H * P1_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&conv2_out, (size_t)eval_bs * C2_OUT * C2_OH * C2_OW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pool2_out, (size_t)eval_bs * FLAT_SIZE * sizeof(float)));

    float *fc1_out, *fc2_out, *out;
    CUDA_CHECK(cudaMalloc(&fc1_out, (size_t)eval_bs * FC1_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fc2_out, (size_t)eval_bs * FC2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out,     (size_t)eval_bs * NUM_CLASSES * sizeof(float)));

    float *d_losses, *d_loss_sum;
    int   *d_correct;
    CUDA_CHECK(cudaMalloc(&d_losses,  (size_t)eval_bs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_correct,  sizeof(int)));

    CUDA_CHECK(cudaMemset(d_loss_sum, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_correct,  0, sizeof(int)));

    int num_batches = (num_samples + eval_bs - 1) / eval_bs;

    for (int batch = 0; batch < num_batches; batch++) {
        int start = batch * eval_bs;
        int bs = eval_bs;
        if (start + bs > num_samples) bs = num_samples - start;

        forward_batch(cnn, d_inputs + start * input_size, bs,
                      conv1_out, pool1_out,
                      conv2_out, pool2_out,
                      fc1_out, fc2_out, out, use_cudnn);

        eval_metrics_kernel<<<BLOCKS(bs), THREADS>>>(
            out, d_targets + start, d_losses, d_correct, bs, NUM_CLASSES);

        cnn_reduce_sum_kernel<<<BLOCKS(bs), THREADS>>>(
            d_losses, d_loss_sum, bs);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    float h_loss;
    int   h_correct;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));

    *loss = h_loss / num_samples;
    *accuracy = (float)h_correct / num_samples * 100.0f;

    cudaFree(d_inputs);   cudaFree(d_targets);
    cudaFree(conv1_out);  cudaFree(pool1_out);
    cudaFree(conv2_out);  cudaFree(pool2_out);
    cudaFree(fc1_out);    cudaFree(fc2_out);   cudaFree(out);
    cudaFree(d_losses);   cudaFree(d_loss_sum); cudaFree(d_correct);
}
