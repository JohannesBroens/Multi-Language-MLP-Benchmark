/*
 * cnn.cu -- CUDA LeNet-5 CNN for MNIST digit classification.
 *
 * Batched convolution: one GEMM per conv layer per batch (not per sample).
 * Channel-major [C, B*spatial] layout through conv layers, transpose at FC boundary.
 * Uses cuBLAS GEMM (via nn_ops) + custom CUDA kernels for elementwise ops.
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
/*  CUDA kernels                                                       */
/* ------------------------------------------------------------------ */

#define THREADS 256
#define BLOCKS(n) (((n) + THREADS - 1) / THREADS)

/* ---- Per-sample im2col/col2im/pool (cnn.h interface) ---- */

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

/* ---- Batched kernels: channel-major [C, B*spatial] layout ---- */

/*
 * Batched im2col: input [C, B*H*W] -> col [K, B*OH*OW]
 * One thread per output element.
 */
__global__ void im2col_batch_kernel(const float *input, float *col,
                                      int C, int B, int H, int W,
                                      int kH, int kW, int stride,
                                      int OH, int OW) {
    int spatial_out = B * OH * OW;
    int K = C * kH * kW;
    int total = K * spatial_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int col_row = idx / spatial_out;
    int col_col = idx % spatial_out;

    /* Decode col_row -> (c, kh_idx, kw_idx) */
    int kw_idx = col_row % kW;
    int tmp = col_row / kW;
    int kh_idx = tmp % kH;
    int c = tmp / kH;

    /* Decode col_col -> (b, oh, ow) */
    int per_sample = OH * OW;
    int b = col_col / per_sample;
    int spatial_idx = col_col % per_sample;
    int oh = spatial_idx / OW;
    int ow = spatial_idx % OW;

    int h = oh * stride + kh_idx;
    int w = ow * stride + kw_idx;

    col[idx] = input[c * B * H * W + b * H * W + h * W + w];
}

/*
 * Batched col2im: col [K, B*OH*OW] -> output [C, B*H*W]  (atomicAdd)
 */
__global__ void col2im_batch_kernel(const float *col, float *output,
                                      int C, int B, int H, int W,
                                      int kH, int kW, int stride,
                                      int OH, int OW) {
    int spatial_out = B * OH * OW;
    int K = C * kH * kW;
    int total = K * spatial_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int col_row = idx / spatial_out;
    int col_col = idx % spatial_out;

    int kw_idx = col_row % kW;
    int tmp = col_row / kW;
    int kh_idx = tmp % kH;
    int c = tmp / kH;

    int per_sample = OH * OW;
    int b = col_col / per_sample;
    int spatial_idx = col_col % per_sample;
    int oh = spatial_idx / OW;
    int ow = spatial_idx % OW;

    int h = oh * stride + kh_idx;
    int w = ow * stride + kw_idx;

    atomicAdd(&output[c * B * H * W + b * H * W + h * W + w], col[idx]);
}

/*
 * Batched avg pool forward: input [C, B*H*W] -> output [C, B*PH*PW]
 * One thread per output element.
 */
__global__ void avg_pool_forward_batch_kernel(const float *input, float *output,
                                                int C, int B, int H, int W,
                                                int pool_size, int PH, int PW) {
    int total = C * B * PH * PW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int pw_idx = idx % PW;
    int tmp = idx / PW;
    int ph_idx = tmp % PH;
    int tmp2 = tmp / PH;
    int b = tmp2 % B;
    int c = tmp2 / B;

    float scale = 1.0f / (pool_size * pool_size);
    float sum = 0.0f;
    for (int i = 0; i < pool_size; i++) {
        for (int j = 0; j < pool_size; j++) {
            int h = ph_idx * pool_size + i;
            int w = pw_idx * pool_size + j;
            sum += input[c * B * H * W + b * H * W + h * W + w];
        }
    }
    output[idx] = sum * scale;
}

/*
 * Batched avg pool backward: grad_output [C, B*PH*PW] -> grad_input [C, B*H*W]
 * One thread per grad_input element.
 */
__global__ void avg_pool_backward_batch_kernel(const float *grad_output, float *grad_input,
                                                 int C, int B, int H, int W,
                                                 int pool_size, int PH, int PW) {
    int total = C * B * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    int tmp2 = tmp / H;
    int b = tmp2 % B;
    int c = tmp2 / B;

    int oh = h / pool_size;
    int ow = w / pool_size;

    if (oh < PH && ow < PW) {
        float scale = 1.0f / (pool_size * pool_size);
        grad_input[idx] = grad_output[c * B * PH * PW + b * PH * PW + oh * PW + ow] * scale;
    }
}

/*
 * Transpose [C, B*S] -> [B, C*S]  (channel-major to batch-major)
 */
__global__ void transpose_cb_to_bc_kernel(const float *src, float *dst,
                                            int C, int B, int S) {
    int total = C * B * S;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    /* src layout: c * B*S + b*S + s */
    int s = idx % S;
    int tmp = idx / S;
    int b = tmp % B;
    int c = tmp / B;

    /* dst layout: b * C*S + c*S + s */
    dst[b * C * S + c * S + s] = src[idx];
}

/*
 * Transpose [B, C*S] -> [C, B*S]  (batch-major to channel-major)
 */
__global__ void transpose_bc_to_cb_kernel(const float *src, float *dst,
                                            int B, int C, int S) {
    int total = B * C * S;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    /* src layout: b * C*S + c*S + s */
    int s = idx % S;
    int tmp = idx / S;
    int c = tmp % C;
    int b = tmp / C;

    /* dst layout: c * B*S + b*S + s */
    dst[c * B * S + b * S + s] = src[idx];
}

/* ---- Existing shared kernels ---- */

__global__ void add_bias_kernel(float *x, const float *bias,
                                 int channels, int spatial) {
    int total = channels * spatial;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int c = idx / spatial;
        x[idx] += bias[c];
    }
}

__global__ void bias_grad_kernel(const float *grad, float *grad_bias,
                                  int channels, int spatial) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float sum = 0.0f;
    const float *gc = grad + c * spatial;
    for (int j = 0; j < spatial; j++)
        sum += gc[j];
    grad_bias[c] = sum;
}

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
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? arr[idx] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
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
    CUDA_CHECK(cudaMallocManaged(&cnn->conv1_weights, C1_OUT * C1_COL_ROWS * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&cnn->conv1_biases,  C1_OUT * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&cnn->conv2_weights, C2_OUT * C2_COL_ROWS * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&cnn->conv2_biases,  C2_OUT * sizeof(float)));

    CUDA_CHECK(cudaMallocManaged(&cnn->fc1_weights, FLAT_SIZE * FC1_OUT * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&cnn->fc1_biases,  FC1_OUT * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&cnn->fc2_weights, FC1_OUT * FC2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&cnn->fc2_biases,  FC2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&cnn->out_weights, FC2_OUT * NUM_CLASSES * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&cnn->out_biases,  NUM_CLASSES * sizeof(float)));

    cudaMemset(cnn->conv1_biases, 0, C1_OUT * sizeof(float));
    cudaMemset(cnn->conv2_biases, 0, C2_OUT * sizeof(float));
    cudaMemset(cnn->fc1_biases,   0, FC1_OUT * sizeof(float));
    cudaMemset(cnn->fc2_biases,   0, FC2_OUT * sizeof(float));
    cudaMemset(cnn->out_biases,   0, NUM_CLASSES * sizeof(float));

    uint32_t rng = (uint32_t)time(NULL);
    nn_xavier_init(cnn->conv1_weights, C1_OUT * C1_COL_ROWS, C1_COL_ROWS, &rng);
    nn_xavier_init(cnn->conv2_weights, C2_OUT * C2_COL_ROWS, C2_COL_ROWS, &rng);
    nn_xavier_init(cnn->fc1_weights, FLAT_SIZE * FC1_OUT, FLAT_SIZE, &rng);
    nn_xavier_init(cnn->fc2_weights, FC1_OUT * FC2_OUT, FC1_OUT, &rng);
    nn_xavier_init(cnn->out_weights, FC2_OUT * NUM_CLASSES, FC2_OUT, &rng);

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
/*  Batched forward pass: one GEMM per conv layer                     */
/* ------------------------------------------------------------------ */

static void forward_batch(const CNN *cnn, const float *d_inputs, int bs,
                          float *col1_b, float *conv1_b, float *pool1_b,
                          float *col2_b, float *conv2_b, float *pool2_b,
                          float *flat,
                          float *fc1_out, float *fc2_out, float *out) {
    int total;

    /* Conv1: inputs[1, B*784] -> col1_b[25, B*576] -> conv1_b[6, B*576] */
    total = C1_COL_ROWS * bs * C1_COL_COLS;
    im2col_batch_kernel<<<BLOCKS(total), THREADS>>>(
        d_inputs, col1_b, C1_IN, bs, IN_H, IN_W, C1_K, C1_K, 1, C1_OH, C1_OW);

    nn_sgemm_nn(C1_OUT, bs * C1_COL_COLS, C1_COL_ROWS,
                cnn->conv1_weights, col1_b, conv1_b);

    total = C1_OUT * bs * C1_COL_COLS;
    add_bias_kernel<<<BLOCKS(total), THREADS>>>(
        conv1_b, cnn->conv1_biases, C1_OUT, bs * C1_COL_COLS);
    nn_relu_forward(conv1_b, total);

    total = C1_OUT * bs * P1_H * P1_W;
    avg_pool_forward_batch_kernel<<<BLOCKS(total), THREADS>>>(
        conv1_b, pool1_b, C1_OUT, bs, C1_OH, C1_OW, P1_SIZE, P1_H, P1_W);

    /* Conv2: pool1_b[6, B*144] -> col2_b[150, B*64] -> conv2_b[16, B*64] */
    total = C2_COL_ROWS * bs * C2_COL_COLS;
    im2col_batch_kernel<<<BLOCKS(total), THREADS>>>(
        pool1_b, col2_b, C2_IN, bs, P1_H, P1_W, C2_K, C2_K, 1, C2_OH, C2_OW);

    nn_sgemm_nn(C2_OUT, bs * C2_COL_COLS, C2_COL_ROWS,
                cnn->conv2_weights, col2_b, conv2_b);

    total = C2_OUT * bs * C2_COL_COLS;
    add_bias_kernel<<<BLOCKS(total), THREADS>>>(
        conv2_b, cnn->conv2_biases, C2_OUT, bs * C2_COL_COLS);
    nn_relu_forward(conv2_b, total);

    total = C2_OUT * bs * P2_H * P2_W;
    avg_pool_forward_batch_kernel<<<BLOCKS(total), THREADS>>>(
        conv2_b, pool2_b, C2_OUT, bs, C2_OH, C2_OW, P2_SIZE, P2_H, P2_W);

    /* Transition: pool2_b[16, B*16] -> flat[B, 256] */
    total = C2_OUT * bs * P2_H * P2_W;
    transpose_cb_to_bc_kernel<<<BLOCKS(total), THREADS>>>(
        pool2_b, flat, C2_OUT, bs, P2_H * P2_W);

    /* FC layers (batch-major, unchanged) */
    nn_sgemm_nn(bs, FC1_OUT, FLAT_SIZE, flat, cnn->fc1_weights, fc1_out);
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
               OptimizerType optimizer, SchedulerType scheduler) {
    (void)optimizer; (void)scheduler; /* TODO: implement in later tasks */

    int input_size = IN_C * IN_H * IN_W;

    /* Copy training data to device */
    float *d_inputs;
    int   *d_targets;
    CUDA_CHECK(cudaMalloc(&d_inputs,  (size_t)num_samples * input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, (size_t)num_samples * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_inputs,  inputs,  (size_t)num_samples * input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets, (size_t)num_samples * sizeof(int), cudaMemcpyHostToDevice));

    /* Forward conv workspace (batched, channel-major) */
    float *col1_b, *conv1_b, *pool1_b, *col2_b, *conv2_b, *pool2_b;
    CUDA_CHECK(cudaMalloc(&col1_b,  (size_t)C1_COL_ROWS * batch_size * C1_COL_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&conv1_b, (size_t)C1_OUT * batch_size * C1_COL_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pool1_b, (size_t)C1_OUT * batch_size * P1_H * P1_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&col2_b,  (size_t)C2_COL_ROWS * batch_size * C2_COL_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&conv2_b, (size_t)C2_OUT * batch_size * C2_COL_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pool2_b, (size_t)C2_OUT * batch_size * P2_H * P2_W * sizeof(float)));

    /* FC layer buffers */
    float *flat, *fc1_out, *fc2_out, *out;
    CUDA_CHECK(cudaMalloc(&flat,    (size_t)batch_size * FLAT_SIZE * sizeof(float)));
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

    /* Conv gradients (computed in one shot via batched GEMM) */
    float *grad_conv2_w, *grad_conv2_b, *grad_conv1_w, *grad_conv1_b;
    CUDA_CHECK(cudaMalloc(&grad_conv2_w, C2_OUT * C2_COL_ROWS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_conv2_b, C2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_conv1_w, C1_OUT * C1_COL_ROWS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_conv1_b, C1_OUT * sizeof(float)));

    /* Backward conv workspace (batched, channel-major) */
    float *d_pool2_b, *d_conv2_b, *d_col2_b, *d_pool1_b, *d_conv1_b;
    CUDA_CHECK(cudaMalloc(&d_pool2_b, (size_t)C2_OUT * batch_size * P2_H * P2_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_b, (size_t)C2_OUT * batch_size * C2_COL_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col2_b,  (size_t)C2_COL_ROWS * batch_size * C2_COL_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool1_b, (size_t)C2_IN * batch_size * P1_H * P1_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_b, (size_t)C1_OUT * batch_size * C1_COL_COLS * sizeof(float)));

    /* Loss tracking */
    float *d_losses, *d_loss_sum;
    CUDA_CHECK(cudaMalloc(&d_losses, (size_t)batch_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_loss_sum, sizeof(float)));

    int num_batches = (num_samples + batch_size - 1) / batch_size;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        *d_loss_sum = 0.0f;

        for (int batch = 0; batch < num_batches; batch++) {
            int start = batch * batch_size;
            int bs = batch_size;
            if (start + bs > num_samples) bs = num_samples - start;

            float *X = d_inputs + start * input_size;
            int   *Y = d_targets + start;

            /* ---- Forward (batched conv + batched FC) ---- */
            forward_batch(cnn, X, bs,
                          col1_b, conv1_b, pool1_b,
                          col2_b, conv2_b, pool2_b,
                          flat, fc1_out, fc2_out, out);

            /* ---- Loss + output gradient ---- */
            nn_cross_entropy_grad(out, Y, d_out, d_losses, bs, NUM_CLASSES);
            cnn_reduce_sum_kernel<<<BLOCKS(bs), THREADS, THREADS * sizeof(float)>>>(
                d_losses, d_loss_sum, bs);

            /* ---- FC backward (unchanged) ---- */
            nn_sgemm_tn(FC2_OUT, NUM_CLASSES, bs, fc2_out, d_out, grad_out_w);
            nn_col_sum(d_out, grad_out_b, bs, NUM_CLASSES);

            nn_sgemm_nt(bs, FC2_OUT, NUM_CLASSES, d_out, cnn->out_weights, d_fc2);
            nn_relu_backward(d_fc2, fc2_out, bs * FC2_OUT);

            nn_sgemm_tn(FC1_OUT, FC2_OUT, bs, fc1_out, d_fc2, grad_fc2_w);
            nn_col_sum(d_fc2, grad_fc2_b, bs, FC2_OUT);

            nn_sgemm_nt(bs, FC1_OUT, FC2_OUT, d_fc2, cnn->fc2_weights, d_fc1);
            nn_relu_backward(d_fc1, fc1_out, bs * FC1_OUT);

            nn_sgemm_tn(FLAT_SIZE, FC1_OUT, bs, flat, d_fc1, grad_fc1_w);
            nn_col_sum(d_fc1, grad_fc1_b, bs, FC1_OUT);

            nn_sgemm_nt(bs, FLAT_SIZE, FC1_OUT, d_fc1, cnn->fc1_weights, d_flat);

            /* ---- Conv backward (batched, one GEMM per layer) ---- */
            int total;

            /* Transition: d_flat[B, 256] -> d_pool2_b[16, B*16] */
            total = bs * FLAT_SIZE;
            transpose_bc_to_cb_kernel<<<BLOCKS(total), THREADS>>>(
                d_flat, d_pool2_b, bs, C2_OUT, P2_H * P2_W);

            /* Pool2 backward */
            total = C2_OUT * bs * C2_OH * C2_OW;
            avg_pool_backward_batch_kernel<<<BLOCKS(total), THREADS>>>(
                d_pool2_b, d_conv2_b, C2_OUT, bs, C2_OH, C2_OW, P2_SIZE, P2_H, P2_W);

            /* ReLU backward conv2 */
            nn_relu_backward(d_conv2_b, conv2_b, C2_OUT * bs * C2_COL_COLS);

            /* Conv2 weight gradient: one GEMM */
            nn_sgemm_nt(C2_OUT, C2_COL_ROWS, bs * C2_COL_COLS,
                        d_conv2_b, col2_b, grad_conv2_w);

            /* Conv2 bias gradient */
            bias_grad_kernel<<<BLOCKS(C2_OUT), THREADS>>>(
                d_conv2_b, grad_conv2_b, C2_OUT, bs * C2_COL_COLS);

            /* Conv2 input gradient */
            nn_sgemm_tn(C2_COL_ROWS, bs * C2_COL_COLS, C2_OUT,
                        cnn->conv2_weights, d_conv2_b, d_col2_b);

            /* col2im batch */
            cudaMemset(d_pool1_b, 0, (size_t)C2_IN * bs * P1_H * P1_W * sizeof(float));
            total = C2_COL_ROWS * bs * C2_COL_COLS;
            col2im_batch_kernel<<<BLOCKS(total), THREADS>>>(
                d_col2_b, d_pool1_b, C2_IN, bs, P1_H, P1_W, C2_K, C2_K, 1, C2_OH, C2_OW);

            /* Pool1 backward */
            total = C1_OUT * bs * C1_OH * C1_OW;
            avg_pool_backward_batch_kernel<<<BLOCKS(total), THREADS>>>(
                d_pool1_b, d_conv1_b, C1_OUT, bs, C1_OH, C1_OW, P1_SIZE, P1_H, P1_W);

            /* ReLU backward conv1 */
            nn_relu_backward(d_conv1_b, conv1_b, C1_OUT * bs * C1_COL_COLS);

            /* Conv1 weight gradient: one GEMM */
            nn_sgemm_nt(C1_OUT, C1_COL_ROWS, bs * C1_COL_COLS,
                        d_conv1_b, col1_b, grad_conv1_w);

            /* Conv1 bias gradient */
            bias_grad_kernel<<<BLOCKS(C1_OUT), THREADS>>>(
                d_conv1_b, grad_conv1_b, C1_OUT, bs * C1_COL_COLS);

            /* ---- SGD update ---- */
            float lr_s = learning_rate / (float)bs;

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

        if (epoch % 1 == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("  Epoch %4d  loss: %.4f\n", epoch, *d_loss_sum / num_samples);
            fflush(stdout);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    /* Free workspace */
    cudaFree(d_inputs);    cudaFree(d_targets);
    cudaFree(col1_b);      cudaFree(conv1_b);    cudaFree(pool1_b);
    cudaFree(col2_b);      cudaFree(conv2_b);    cudaFree(pool2_b);
    cudaFree(flat);        cudaFree(fc1_out);    cudaFree(fc2_out);    cudaFree(out);
    cudaFree(d_out);       cudaFree(d_fc2);      cudaFree(d_fc1);      cudaFree(d_flat);
    cudaFree(grad_out_w);  cudaFree(grad_out_b);
    cudaFree(grad_fc2_w);  cudaFree(grad_fc2_b);
    cudaFree(grad_fc1_w);  cudaFree(grad_fc1_b);
    cudaFree(grad_conv2_w); cudaFree(grad_conv2_b);
    cudaFree(grad_conv1_w); cudaFree(grad_conv1_b);
    cudaFree(d_pool2_b);   cudaFree(d_conv2_b);  cudaFree(d_col2_b);
    cudaFree(d_pool1_b);   cudaFree(d_conv1_b);
    cudaFree(d_losses);    cudaFree(d_loss_sum);
}

/* ------------------------------------------------------------------ */
/*  Evaluation                                                         */
/* ------------------------------------------------------------------ */

void cnn_evaluate(CNN *cnn, float *inputs, int *targets, int num_samples,
                  float *loss, float *accuracy) {
    int input_size = IN_C * IN_H * IN_W;
    int eval_bs = 256;

    /* Copy data to device */
    float *d_inputs;
    int   *d_targets;
    CUDA_CHECK(cudaMalloc(&d_inputs,  (size_t)num_samples * input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, (size_t)num_samples * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_inputs,  inputs,  (size_t)num_samples * input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets, (size_t)num_samples * sizeof(int), cudaMemcpyHostToDevice));

    /* Batched conv workspace */
    float *col1_b, *conv1_b, *pool1_b, *col2_b, *conv2_b, *pool2_b;
    CUDA_CHECK(cudaMalloc(&col1_b,  (size_t)C1_COL_ROWS * eval_bs * C1_COL_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&conv1_b, (size_t)C1_OUT * eval_bs * C1_COL_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pool1_b, (size_t)C1_OUT * eval_bs * P1_H * P1_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&col2_b,  (size_t)C2_COL_ROWS * eval_bs * C2_COL_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&conv2_b, (size_t)C2_OUT * eval_bs * C2_COL_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pool2_b, (size_t)C2_OUT * eval_bs * P2_H * P2_W * sizeof(float)));

    float *flat, *fc1_out, *fc2_out, *out;
    CUDA_CHECK(cudaMalloc(&flat,    (size_t)eval_bs * FLAT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fc1_out, (size_t)eval_bs * FC1_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fc2_out, (size_t)eval_bs * FC2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out,     (size_t)eval_bs * NUM_CLASSES * sizeof(float)));

    float *d_losses, *d_loss_sum;
    int   *d_correct;
    CUDA_CHECK(cudaMalloc(&d_losses, (size_t)eval_bs * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_loss_sum, sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_correct,  sizeof(int)));

    *d_loss_sum = 0.0f;
    *d_correct  = 0;

    int num_batches = (num_samples + eval_bs - 1) / eval_bs;

    for (int batch = 0; batch < num_batches; batch++) {
        int start = batch * eval_bs;
        int bs = eval_bs;
        if (start + bs > num_samples) bs = num_samples - start;

        forward_batch(cnn, d_inputs + start * input_size, bs,
                      col1_b, conv1_b, pool1_b,
                      col2_b, conv2_b, pool2_b,
                      flat, fc1_out, fc2_out, out);

        eval_metrics_kernel<<<BLOCKS(bs), THREADS>>>(
            out, d_targets + start, d_losses, d_correct, bs, NUM_CLASSES);

        cnn_reduce_sum_kernel<<<BLOCKS(bs), THREADS, THREADS * sizeof(float)>>>(
            d_losses, d_loss_sum, bs);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    *loss = *d_loss_sum / num_samples;
    *accuracy = (float)(*d_correct) / num_samples * 100.0f;

    cudaFree(d_inputs);   cudaFree(d_targets);
    cudaFree(col1_b);     cudaFree(conv1_b);   cudaFree(pool1_b);
    cudaFree(col2_b);     cudaFree(conv2_b);   cudaFree(pool2_b);
    cudaFree(flat);       cudaFree(fc1_out);   cudaFree(fc2_out);   cudaFree(out);
    cudaFree(d_losses);   cudaFree(d_loss_sum); cudaFree(d_correct);
}
