/*
 * nn_ops.cu -- CUDA implementation of shared neural network operations.
 *
 * Uses cuBLAS for GEMM (row-major via transposition trick) and custom
 * CUDA kernels for elementwise ops (relu, softmax, cross-entropy, SGD).
 *
 * The same nn_ops.h interface is preserved so that models (MLP, CNN)
 * link against nn_ops regardless of CPU or CUDA build.
 */

#include "nn_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = (call); \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                __FILE__, __LINE__, stat); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/* ------------------------------------------------------------------ */
/*  Persistent cuBLAS handle, lazily initialized                      */
/* ------------------------------------------------------------------ */

static cublasHandle_t g_cublas_handle = NULL;

static void ensure_cublas(void) {
    if (!g_cublas_handle)
        CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
}

/* ------------------------------------------------------------------ */
/*  cuBLAS GEMM wrappers (row-major via transposition trick)          */
/*  Row-major C[M,N] = A[M,K] @ B[K,N]                               */
/*  In col-major: C' = B' @ A'  (swap A<->B, swap M<->N)             */
/* ------------------------------------------------------------------ */

void nn_sgemm_nn(int M, int N, int K,
                 const float *A, const float *B, float *C) {
    ensure_cublas();
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

void nn_sgemm_tn(int M, int N, int K,
                 const float *A, const float *B, float *C) {
    ensure_cublas();
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             N, M, K, &alpha, B, N, A, M, &beta, C, N));
}

void nn_sgemm_nt(int M, int N, int K,
                 const float *A, const float *B, float *C) {
    ensure_cublas();
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K, &alpha, B, K, A, K, &beta, C, N));
}

/* ------------------------------------------------------------------ */
/*  CUDA kernels for elementwise operations                           */
/* ------------------------------------------------------------------ */

__global__ void relu_forward_kernel(float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        if (x[idx] < 0.0f) x[idx] = 0.0f;
}

__global__ void relu_backward_kernel(float *grad, const float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        if (x[idx] <= 0.0f) grad[idx] = 0.0f;
}

__global__ void bias_relu_kernel(float *x, const float *bias, int rows, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * width;
    if (idx < total) {
        float v = x[idx] + bias[idx % width];
        x[idx] = (v > 0.0f) ? v : 0.0f;
    }
}

__global__ void softmax_kernel(float *x, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float *xi = x + row * cols;
    float mx = -1e30f;
    for (int j = 0; j < cols; ++j)
        if (xi[j] > mx) mx = xi[j];

    float s = 0.0f;
    for (int j = 0; j < cols; ++j) {
        xi[j] = expf(xi[j] - mx);
        s += xi[j];
    }
    for (int j = 0; j < cols; ++j)
        xi[j] /= s;
}

__global__ void bias_softmax_kernel(float *x, const float *bias, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float *xi = x + row * cols;
    float mx = -1e30f;
    for (int j = 0; j < cols; ++j) {
        xi[j] += bias[j];
        if (xi[j] > mx) mx = xi[j];
    }

    float s = 0.0f;
    for (int j = 0; j < cols; ++j) {
        xi[j] = expf(xi[j] - mx);
        s += xi[j];
    }
    for (int j = 0; j < cols; ++j)
        xi[j] /= s;
}

__global__ void cross_entropy_grad_kernel(const float *probs, const int *targets,
                                           float *grad, float *losses,
                                           int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float *p = probs + row * cols;
    float *g = grad + row * cols;
    int t = targets[row];

    if (losses)
        losses[row] = -logf(p[t] + 1e-7f);

    for (int j = 0; j < cols; ++j)
        g[j] = p[j] - ((t == j) ? 1.0f : 0.0f);
}

__global__ void col_sum_kernel(const float *mat, float *out, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float s = 0.0f;
    for (int i = 0; i < rows; ++i)
        s += mat[i * cols + j];
    out[j] = s;
}

__global__ void sgd_kernel(float *param, const float *grad, float lr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        param[idx] -= lr * grad[idx];
}

/* Reduce sum via shared memory + atomicAdd */
__global__ void reduce_sum_kernel(const float *arr, float *result, int n) {
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
/*  nn_ops.h interface implementations (CUDA device-pointer versions) */
/* ------------------------------------------------------------------ */

#define THREADS 256
#define BLOCKS(n) (((n) + THREADS - 1) / THREADS)

void nn_relu_forward(float *x, int n) {
    relu_forward_kernel<<<BLOCKS(n), THREADS>>>(x, n);
}

void nn_relu_backward(float *grad, const float *x, int n) {
    relu_backward_kernel<<<BLOCKS(n), THREADS>>>(grad, x, n);
}

void nn_bias_relu(float *x, const float *bias, int rows, int width) {
    int total = rows * width;
    bias_relu_kernel<<<BLOCKS(total), THREADS>>>(x, bias, rows, width);
}

void nn_softmax(float *x, int rows, int cols) {
    softmax_kernel<<<BLOCKS(rows), THREADS>>>(x, rows, cols);
}

void nn_bias_softmax(float *x, const float *bias, int rows, int cols) {
    bias_softmax_kernel<<<BLOCKS(rows), THREADS>>>(x, bias, rows, cols);
}

float nn_cross_entropy_loss(const float *probs, const int *targets, int rows, int cols) {
    /* Allocate temp losses + sum on device */
    float *d_losses, *d_sum;
    CUDA_CHECK(cudaMalloc(&d_losses, rows * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_sum, sizeof(float)));
    *d_sum = 0.0f;

    /* Compute per-sample loss */
    cross_entropy_grad_kernel<<<BLOCKS(rows), THREADS>>>(
        probs, targets, NULL, d_losses, rows, cols);

    reduce_sum_kernel<<<BLOCKS(rows), THREADS, THREADS * sizeof(float)>>>(
        d_losses, d_sum, rows);

    CUDA_CHECK(cudaDeviceSynchronize());
    float result = *d_sum / rows;

    cudaFree(d_losses);
    cudaFree(d_sum);
    return result;
}

void nn_cross_entropy_grad(const float *probs, const int *targets,
                           float *grad, float *losses, int rows, int cols) {
    cross_entropy_grad_kernel<<<BLOCKS(rows), THREADS>>>(
        probs, targets, grad, losses, rows, cols);
}

void nn_col_sum(const float *mat, float *out, int rows, int cols) {
    col_sum_kernel<<<BLOCKS(cols), THREADS>>>(mat, out, rows, cols);
}

void nn_sgd_update(float *param, const float *grad, float lr, int n) {
    sgd_kernel<<<BLOCKS(n), THREADS>>>(param, grad, lr, n);
}

/* ------------------------------------------------------------------ */
/*  Learning rate scheduler -- host-side only                          */
/* ------------------------------------------------------------------ */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float nn_cosine_lr(int epoch, int total_epochs, float lr_max,
                   int warmup_epochs, float lr_min) {
    if (epoch < warmup_epochs) {
        return lr_max * ((float)(epoch + 1) / (float)warmup_epochs);
    }
    float progress = (float)(epoch - warmup_epochs) / (float)(total_epochs - warmup_epochs);
    return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf((float)M_PI * progress));
}

/* ------------------------------------------------------------------ */
/*  PRNG -- host-side only (used for weight init before copy to GPU)  */
/* ------------------------------------------------------------------ */

unsigned int nn_xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

float nn_rand_float(unsigned int *state) {
    return (float)(nn_xorshift32(state) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

void nn_xavier_init(float *weights, int size, int fan_in, unsigned int *rng) {
    float scale = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < size; ++i)
        weights[i] = (nn_rand_float(rng) * 2.0f - 1.0f) * scale;
}

/* No physical core detection needed for CUDA */
int nn_detect_physical_cores(void) {
    return 1;
}
