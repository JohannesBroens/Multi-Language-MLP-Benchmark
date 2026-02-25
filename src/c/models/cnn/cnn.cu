/*
 * cnn.cu -- CUDA LeNet-5 CNN for MNIST digit classification.
 *
 * Uses cuBLAS GEMM (via nn_ops) for convolution (im2col->GEMM) and FC layers,
 * plus custom CUDA kernels for im2col, col2im, avg_pool, and bias ops.
 * All data resides on device; only final metrics are copied to host.
 *
 * Architecture: Conv1(5x5,6) -> ReLU -> Pool(2x2) -> Conv2(5x5,16) -> ReLU -> Pool(2x2)
 *             -> Flatten(256) -> FC1(120) -> ReLU -> FC2(84) -> ReLU -> Output(10) -> Softmax
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
/*  CUDA kernels: im2col, col2im, avg_pool, add_bias                  */
/* ------------------------------------------------------------------ */

#define THREADS 256
#define BLOCKS(n) (((n) + THREADS - 1) / THREADS)

/* im2col kernel: one thread per output element.
 * Output shape: [C*kH*kW, OH*OW] */
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

    /* Decode col_row -> (c, kh, kw) */
    int kw_idx = col_row % kW;
    int tmp = col_row / kW;
    int kh_idx = tmp % kH;
    int c = tmp / kH;

    /* Decode col_col -> (oh, ow) */
    int oh = col_col / OW;
    int ow = col_col % OW;

    int h = oh * stride + kh_idx;
    int w = ow * stride + kw_idx;

    col[idx] = input[c * H * W + h * W + w];
}

/* col2im kernel: one thread per col element, atomicAdd for overlapping patches */
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

/* Average pooling forward: one thread per output element */
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

/* Average pooling backward: one thread per output (grad_input) element */
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

/* Add bias to [channels, spatial] matrix: x[c*spatial + j] += bias[c] */
__global__ void add_bias_kernel(float *x, const float *bias,
                                 int channels, int spatial) {
    int total = channels * spatial;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int c = idx / spatial;
        x[idx] += bias[c];
    }
}

/* Bias gradient for conv layers: sum spatial dims per channel */
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

/* Eval metrics: compute loss + argmax per sample */
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

/* Reduce sum via shared memory */
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
/*  im2col / col2im / avg_pool C interface (used by cnn.h)            */
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
/*  Workspace layout for per-sample conv forward pass                 */
/* ------------------------------------------------------------------ */

#define WS_COL1     0
#define WS_CONV1    (C1_COL_ROWS * C1_COL_COLS)
#define WS_POOL1    (WS_CONV1 + C1_OUT * C1_COL_COLS)
#define WS_COL2     (WS_POOL1 + C1_OUT * P1_H * P1_W)
#define WS_CONV2    (WS_COL2 + C2_COL_ROWS * C2_COL_COLS)
#define WS_POOL2    (WS_CONV2 + C2_OUT * C2_COL_COLS)
#define WS_SIZE     (WS_POOL2 + FLAT_SIZE)

/* ------------------------------------------------------------------ */
/*  Initialize / Free -- managed memory for parameters                */
/* ------------------------------------------------------------------ */

void cnn_initialize(CNN *cnn) {
    /* Allocate on host, Xavier init, then copy to device would be cleaner,
     * but using managed memory keeps it simple and matches mlp.cu pattern */
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

    /* Zero biases */
    cudaMemset(cnn->conv1_biases, 0, C1_OUT * sizeof(float));
    cudaMemset(cnn->conv2_biases, 0, C2_OUT * sizeof(float));
    cudaMemset(cnn->fc1_biases,   0, FC1_OUT * sizeof(float));
    cudaMemset(cnn->fc2_biases,   0, FC2_OUT * sizeof(float));
    cudaMemset(cnn->out_biases,   0, NUM_CLASSES * sizeof(float));

    /* Xavier init on host (managed memory is host-accessible) */
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
/*  Per-sample conv forward pass on GPU                               */
/* ------------------------------------------------------------------ */

static void conv_forward_sample(const CNN *cnn, const float *input, float *ws) {
    float *col1      = ws + WS_COL1;
    float *conv1_out = ws + WS_CONV1;
    float *pool1     = ws + WS_POOL1;
    float *col2      = ws + WS_COL2;
    float *conv2_out = ws + WS_CONV2;
    float *pool2     = ws + WS_POOL2;

    /* Conv1: im2col -> GEMM -> bias + ReLU -> pool */
    im2col(input, C1_IN, IN_H, IN_W, C1_K, C1_K, 1, col1);
    nn_sgemm_nn(C1_OUT, C1_COL_COLS, C1_COL_ROWS,
                cnn->conv1_weights, col1, conv1_out);
    add_bias_kernel<<<BLOCKS(C1_OUT * C1_COL_COLS), THREADS>>>(
        conv1_out, cnn->conv1_biases, C1_OUT, C1_COL_COLS);
    nn_relu_forward(conv1_out, C1_OUT * C1_COL_COLS);
    avg_pool_forward(conv1_out, pool1, C1_OUT, C1_OH, C1_OW, P1_SIZE);

    /* Conv2: im2col -> GEMM -> bias + ReLU -> pool */
    im2col(pool1, C2_IN, P1_H, P1_W, C2_K, C2_K, 1, col2);
    nn_sgemm_nn(C2_OUT, C2_COL_COLS, C2_COL_ROWS,
                cnn->conv2_weights, col2, conv2_out);
    add_bias_kernel<<<BLOCKS(C2_OUT * C2_COL_COLS), THREADS>>>(
        conv2_out, cnn->conv2_biases, C2_OUT, C2_COL_COLS);
    nn_relu_forward(conv2_out, C2_OUT * C2_COL_COLS);
    avg_pool_forward(conv2_out, pool2, C2_OUT, C2_OH, C2_OW, P2_SIZE);
}

/* Copy kernel: copy pool2 from per-sample workspace to flat batch matrix */
__global__ void copy_flat_kernel(const float *ws_all, float *flat,
                                  int ws_size, int ws_pool2_offset,
                                  int flat_size, int bs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bs * flat_size) return;
    int b = idx / flat_size;
    int j = idx % flat_size;
    flat[idx] = ws_all[b * ws_size + ws_pool2_offset + j];
}

/* ------------------------------------------------------------------ */
/*  Batched forward pass                                              */
/* ------------------------------------------------------------------ */

static void forward_batch(const CNN *cnn, const float *d_inputs, int bs,
                          float *ws_all, float *flat,
                          float *fc1_out, float *fc2_out, float *out) {
    int input_size = IN_C * IN_H * IN_W;

    /* Per-sample conv layers */
    for (int b = 0; b < bs; b++) {
        const float *img = d_inputs + b * input_size;
        float *ws = ws_all + b * WS_SIZE;
        conv_forward_sample(cnn, img, ws);
    }

    /* Copy pool2 output to flat batch matrix */
    int total_flat = bs * FLAT_SIZE;
    copy_flat_kernel<<<BLOCKS(total_flat), THREADS>>>(
        ws_all, flat, WS_SIZE, WS_POOL2, FLAT_SIZE, bs);

    /* FC1: flat[bs, 256] @ fc1_weights[256, 120] -> fc1_out[bs, 120] */
    nn_sgemm_nn(bs, FC1_OUT, FLAT_SIZE, flat, cnn->fc1_weights, fc1_out);
    nn_bias_relu(fc1_out, cnn->fc1_biases, bs, FC1_OUT);

    /* FC2: fc1_out[bs, 120] @ fc2_weights[120, 84] -> fc2_out[bs, 84] */
    nn_sgemm_nn(bs, FC2_OUT, FC1_OUT, fc1_out, cnn->fc2_weights, fc2_out);
    nn_bias_relu(fc2_out, cnn->fc2_biases, bs, FC2_OUT);

    /* Output: fc2_out[bs, 84] @ out_weights[84, 10] -> out[bs, 10] */
    nn_sgemm_nn(bs, NUM_CLASSES, FC2_OUT, fc2_out, cnn->out_weights, out);
    nn_bias_softmax(out, cnn->out_biases, bs, NUM_CLASSES);
}

/* ------------------------------------------------------------------ */
/*  Training                                                           */
/* ------------------------------------------------------------------ */

void cnn_train(CNN *cnn, float *inputs, int *targets, int num_samples,
               int batch_size, int num_epochs, float learning_rate) {

    int input_size = IN_C * IN_H * IN_W;

    /* Copy training data to device */
    float *d_inputs;
    int   *d_targets;
    CUDA_CHECK(cudaMalloc(&d_inputs,  (size_t)num_samples * input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, (size_t)num_samples * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_inputs,  inputs,  (size_t)num_samples * input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets, (size_t)num_samples * sizeof(int), cudaMemcpyHostToDevice));

    /* Forward workspace (per sample conv buffers) */
    float *ws_all;
    CUDA_CHECK(cudaMalloc(&ws_all, (size_t)batch_size * WS_SIZE * sizeof(float)));

    /* FC layer outputs */
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

    /* Conv gradients */
    float *grad_conv2_w, *grad_conv2_b, *grad_conv1_w, *grad_conv1_b;
    CUDA_CHECK(cudaMalloc(&grad_conv2_w, C2_OUT * C2_COL_ROWS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_conv2_b, C2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_conv1_w, C1_OUT * C1_COL_ROWS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad_conv1_b, C1_OUT * sizeof(float)));

    /* Per-sample backward workspace for conv */
    float *d_pool2, *d_conv2_out, *d_col2, *d_pool1, *d_conv1_out;
    CUDA_CHECK(cudaMalloc(&d_pool2,     FLAT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_out, C2_OUT * C2_COL_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col2,      C2_COL_ROWS * C2_COL_COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool1,     C2_IN * P1_H * P1_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv1_out, C1_OUT * C1_COL_COLS * sizeof(float)));

    /* Temp for per-sample conv weight and bias gradients */
    float *tmp_grad_w2, *tmp_grad_w1;
    float *tmp_grad_b2, *tmp_grad_b1;
    CUDA_CHECK(cudaMalloc(&tmp_grad_w2, C2_OUT * C2_COL_ROWS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tmp_grad_w1, C1_OUT * C1_COL_ROWS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tmp_grad_b2, C2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&tmp_grad_b1, C1_OUT * sizeof(float)));

    /* Loss tracking */
    float *d_losses, *d_loss_sum;
    CUDA_CHECK(cudaMalloc(&d_losses, (size_t)batch_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_loss_sum, sizeof(float)));

    int num_batches = (num_samples + batch_size - 1) / batch_size;

    /* --- Accumulation kernel for adding per-sample conv grads --- */
    /* We use nn_sgd_update with lr=-1 as an axpy to accumulate:
     * grad_conv_w += tmp_grad_w   =>   grad_conv_w -= (-1) * tmp_grad_w */

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        *d_loss_sum = 0.0f;

        for (int batch = 0; batch < num_batches; batch++) {
            int start = batch * batch_size;
            int bs = batch_size;
            if (start + bs > num_samples) bs = num_samples - start;

            float *X = d_inputs + start * input_size;
            int   *Y = d_targets + start;

            /* ---- Forward ---- */
            forward_batch(cnn, X, bs, ws_all, flat, fc1_out, fc2_out, out);

            /* ---- Loss + output gradient ---- */
            nn_cross_entropy_grad(out, Y, d_out, d_losses, bs, NUM_CLASSES);

            /* Accumulate batch loss */
            cnn_reduce_sum_kernel<<<BLOCKS(bs), THREADS, THREADS * sizeof(float)>>>(
                d_losses, d_loss_sum, bs);

            /* ---- FC backward ---- */

            /* grad_out_w = fc2_out^T @ d_out */
            nn_sgemm_tn(FC2_OUT, NUM_CLASSES, bs, fc2_out, d_out, grad_out_w);
            nn_col_sum(d_out, grad_out_b, bs, NUM_CLASSES);

            /* d_fc2 = d_out @ out_weights^T, then ReLU backward */
            nn_sgemm_nt(bs, FC2_OUT, NUM_CLASSES, d_out, cnn->out_weights, d_fc2);
            nn_relu_backward(d_fc2, fc2_out, bs * FC2_OUT);

            nn_sgemm_tn(FC1_OUT, FC2_OUT, bs, fc1_out, d_fc2, grad_fc2_w);
            nn_col_sum(d_fc2, grad_fc2_b, bs, FC2_OUT);

            nn_sgemm_nt(bs, FC1_OUT, FC2_OUT, d_fc2, cnn->fc2_weights, d_fc1);
            nn_relu_backward(d_fc1, fc1_out, bs * FC1_OUT);

            nn_sgemm_tn(FLAT_SIZE, FC1_OUT, bs, flat, d_fc1, grad_fc1_w);
            nn_col_sum(d_fc1, grad_fc1_b, bs, FC1_OUT);

            /* d_flat = d_fc1 @ fc1_weights^T */
            nn_sgemm_nt(bs, FLAT_SIZE, FC1_OUT, d_fc1, cnn->fc1_weights, d_flat);

            /* ---- Conv backward (per-sample) ---- */
            cudaMemset(grad_conv2_w, 0, C2_OUT * C2_COL_ROWS * sizeof(float));
            cudaMemset(grad_conv2_b, 0, C2_OUT * sizeof(float));
            cudaMemset(grad_conv1_w, 0, C1_OUT * C1_COL_ROWS * sizeof(float));
            cudaMemset(grad_conv1_b, 0, C1_OUT * sizeof(float));

            for (int b = 0; b < bs; b++) {
                float *ws = ws_all + b * WS_SIZE;
                float *col1      = ws + WS_COL1;
                float *conv1_out = ws + WS_CONV1;
                float *col2      = ws + WS_COL2;
                float *conv2_out = ws + WS_CONV2;

                /* Copy d_flat[b] to d_pool2 */
                CUDA_CHECK(cudaMemcpy(d_pool2, d_flat + b * FLAT_SIZE,
                                      FLAT_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));

                /* Pool2 backward */
                cudaMemset(d_conv2_out, 0, C2_OUT * C2_COL_COLS * sizeof(float));
                avg_pool_backward(d_pool2, d_conv2_out, C2_OUT, C2_OH, C2_OW, P2_SIZE);

                /* ReLU backward for conv2 */
                nn_relu_backward(d_conv2_out, conv2_out, C2_OUT * C2_COL_COLS);

                /* Conv2 weight gradient: d_conv2_out[16,64] @ col2^T[64,150] */
                nn_sgemm_nt(C2_OUT, C2_COL_ROWS, C2_COL_COLS,
                            d_conv2_out, col2, tmp_grad_w2);
                /* Accumulate: grad_conv2_w += tmp_grad_w2 */
                nn_sgd_update(grad_conv2_w, tmp_grad_w2, -1.0f, C2_OUT * C2_COL_ROWS);

                /* Conv2 bias gradient */
                bias_grad_kernel<<<BLOCKS(C2_OUT), THREADS>>>(
                    d_conv2_out, tmp_grad_b2, C2_OUT, C2_COL_COLS);
                nn_sgd_update(grad_conv2_b, tmp_grad_b2, -1.0f, C2_OUT);

                /* Conv2 input gradient: conv2_weights^T @ d_conv2_out -> d_col2 */
                nn_sgemm_tn(C2_COL_ROWS, C2_COL_COLS, C2_OUT,
                            cnn->conv2_weights, d_conv2_out, d_col2);

                /* col2im -> d_pool1 */
                cudaMemset(d_pool1, 0, C2_IN * P1_H * P1_W * sizeof(float));
                col2im(d_col2, C2_IN, P1_H, P1_W, C2_K, C2_K, 1, d_pool1);

                /* Pool1 backward */
                cudaMemset(d_conv1_out, 0, C1_OUT * C1_COL_COLS * sizeof(float));
                avg_pool_backward(d_pool1, d_conv1_out, C1_OUT, C1_OH, C1_OW, P1_SIZE);

                /* ReLU backward for conv1 */
                nn_relu_backward(d_conv1_out, conv1_out, C1_OUT * C1_COL_COLS);

                /* Conv1 weight gradient */
                nn_sgemm_nt(C1_OUT, C1_COL_ROWS, C1_COL_COLS,
                            d_conv1_out, col1, tmp_grad_w1);
                nn_sgd_update(grad_conv1_w, tmp_grad_w1, -1.0f, C1_OUT * C1_COL_ROWS);

                /* Conv1 bias gradient */
                bias_grad_kernel<<<BLOCKS(C1_OUT), THREADS>>>(
                    d_conv1_out, tmp_grad_b1, C1_OUT, C1_COL_COLS);
                nn_sgd_update(grad_conv1_b, tmp_grad_b1, -1.0f, C1_OUT);
            }

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
    cudaFree(ws_all);      cudaFree(flat);
    cudaFree(fc1_out);     cudaFree(fc2_out);    cudaFree(out);
    cudaFree(d_out);       cudaFree(d_fc2);      cudaFree(d_fc1);      cudaFree(d_flat);
    cudaFree(grad_out_w);  cudaFree(grad_out_b);
    cudaFree(grad_fc2_w);  cudaFree(grad_fc2_b);
    cudaFree(grad_fc1_w);  cudaFree(grad_fc1_b);
    cudaFree(grad_conv2_w); cudaFree(grad_conv2_b);
    cudaFree(grad_conv1_w); cudaFree(grad_conv1_b);
    cudaFree(d_pool2);     cudaFree(d_conv2_out); cudaFree(d_col2);
    cudaFree(d_pool1);     cudaFree(d_conv1_out);
    cudaFree(tmp_grad_w2); cudaFree(tmp_grad_w1);
    cudaFree(tmp_grad_b2); cudaFree(tmp_grad_b1);
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

    float *ws_all, *flat, *fc1_out, *fc2_out, *out;
    CUDA_CHECK(cudaMalloc(&ws_all,  (size_t)eval_bs * WS_SIZE * sizeof(float)));
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
                      ws_all, flat, fc1_out, fc2_out, out);

        eval_metrics_kernel<<<BLOCKS(bs), THREADS>>>(
            out, d_targets + start, d_losses, d_correct, bs, NUM_CLASSES);

        cnn_reduce_sum_kernel<<<BLOCKS(bs), THREADS, THREADS * sizeof(float)>>>(
            d_losses, d_loss_sum, bs);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    *loss = *d_loss_sum / num_samples;
    *accuracy = (float)(*d_correct) / num_samples * 100.0f;

    cudaFree(d_inputs);   cudaFree(d_targets);
    cudaFree(ws_all);     cudaFree(flat);
    cudaFree(fc1_out);    cudaFree(fc2_out);    cudaFree(out);
    cudaFree(d_losses);   cudaFree(d_loss_sum); cudaFree(d_correct);
}
