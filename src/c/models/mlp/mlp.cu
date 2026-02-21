#include "mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = (call); \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, stat); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/* Persistent cuBLAS handle */
static cublasHandle_t cublas_handle = NULL;

/* ------------------------------------------------------------------ */
/*  cuBLAS GEMM wrappers for row-major data                          */
/*  Row-major C[M,N] = A[M,K] @ B[K,N] uses the transposition trick: */
/*  In col-major: C' = B' @ A'  (swap A and B, swap M and N)         */
/* ------------------------------------------------------------------ */

/* C[M,N] = A[M,K] @ B[K,N] */
static void sgemm_nn(int M, int N, int K,
                     const float *A, const float *B, float *C) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

/* C[M,N] = A^T[M,K] @ B[K,N],  A stored as [K,M] */
static void sgemm_tn(int M, int N, int K,
                     const float *A, const float *B, float *C) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             N, M, K, &alpha, B, N, A, M, &beta, C, N));
}

/* C[M,N] = A[M,K] @ B^T[K,N],  B stored as [N,K] */
static void sgemm_nt(int M, int N, int K,
                     const float *A, const float *B, float *C) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K, &alpha, B, K, A, K, &beta, C, N));
}

/* ------------------------------------------------------------------ */
/*  Element-wise CUDA kernels                                        */
/* ------------------------------------------------------------------ */

/* Add bias vector then apply ReLU: hidden[i] = max(0, hidden[i] + bias[i % hid]) */
__global__ void bias_relu_kernel(float *hidden, const float *bias, int total, int hid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float v = hidden[idx] + bias[idx % hid];
        hidden[idx] = (v > 0.0f) ? v : 0.0f;
    }
}

/* Add bias then numerically stable softmax per row */
__global__ void bias_softmax_kernel(float *output, const float *bias, int bs, int out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= bs) return;

    float *o = output + row * out;

    /* Add bias and find max */
    float mx = -1e30f;
    for (int j = 0; j < out; ++j) {
        o[j] += bias[j];
        if (o[j] > mx) mx = o[j];
    }

    /* Exp and sum */
    float s = 0.0f;
    for (int j = 0; j < out; ++j) {
        o[j] = expf(o[j] - mx);
        s += o[j];
    }

    /* Normalize */
    for (int j = 0; j < out; ++j)
        o[j] /= s;
}

/* d_output = output - one_hot(targets); also accumulates loss */
__global__ void ce_grad_kernel(const float *output, const int *targets,
                               float *d_output, float *losses,
                               int bs, int out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= bs) return;

    const float *o = output  + row * out;
    float       *d = d_output + row * out;
    int t = targets[row];

    losses[row] = -logf(o[t] + 1e-7f);

    for (int j = 0; j < out; ++j)
        d[j] = o[j] - ((t == j) ? 1.0f : 0.0f);
}

/* Apply ReLU mask: d_hidden[i] = (hidden[i] > 0) ? d_hidden[i] : 0 */
__global__ void relu_mask_kernel(float *d_hidden, const float *hidden, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
        if (hidden[idx] <= 0.0f) d_hidden[idx] = 0.0f;
}

/* Column sum: out_vec[j] = sum_i(mat[i * cols + j]) */
__global__ void col_sum_kernel(const float *mat, float *out_vec, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float s = 0.0f;
    for (int i = 0; i < rows; ++i)
        s += mat[i * cols + j];
    out_vec[j] = s;
}

/* SGD update: param[i] -= lr_scaled * grad[i] */
__global__ void sgd_kernel(float *param, const float *grad, float lr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        param[idx] -= lr * grad[idx];
}

/* Reduce sum of a float array */
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
/*  Xavier init kernel                                                */
/* ------------------------------------------------------------------ */

__global__ void init_weights_kernel(float *weights, int size, float scale, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        weights[idx] = (curand_uniform(&state) * 2.0f - 1.0f) * scale;
    }
}

/* ------------------------------------------------------------------ */
/*  Init / Free                                                       */
/* ------------------------------------------------------------------ */

void mlp_initialize(MLP *mlp, int input_size, int hidden_size, int output_size) {
    mlp->input_size = input_size;
    mlp->hidden_size = hidden_size;
    mlp->output_size = output_size;

    /* Unified memory for weights — accessible from host and device */
    CUDA_CHECK(cudaMallocManaged(&mlp->input_weights,  input_size  * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&mlp->output_weights, hidden_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&mlp->hidden_biases,  hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&mlp->output_biases,  output_size * sizeof(float)));

    cudaMemset(mlp->hidden_biases, 0, hidden_size * sizeof(float));
    cudaMemset(mlp->output_biases, 0, output_size * sizeof(float));

    int threads = 256;
    unsigned long seed = (unsigned long)time(NULL);

    float scale_ih = sqrtf(2.0f / (float)input_size);
    int blocks_ih = (input_size * hidden_size + threads - 1) / threads;
    init_weights_kernel<<<blocks_ih, threads>>>(mlp->input_weights,
                                                 input_size * hidden_size, scale_ih, seed);

    float scale_ho = sqrtf(2.0f / (float)hidden_size);
    int blocks_ho = (hidden_size * output_size + threads - 1) / threads;
    init_weights_kernel<<<blocks_ho, threads>>>(mlp->output_weights,
                                                 hidden_size * output_size, scale_ho, seed + 1);

    CUDA_CHECK(cudaDeviceSynchronize());

    /* Create cuBLAS handle */
    if (!cublas_handle)
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
}

void mlp_free(MLP *mlp) {
    cudaFree(mlp->input_weights);
    cudaFree(mlp->output_weights);
    cudaFree(mlp->hidden_biases);
    cudaFree(mlp->output_biases);

    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = NULL;
    }
}

/* ------------------------------------------------------------------ */
/*  Training: cuBLAS GEMM + elementwise kernels                      */
/* ------------------------------------------------------------------ */

void mlp_train(MLP *mlp, float *inputs, int *targets, int num_samples,
               int batch_size, int num_epochs, float learning_rate) {
    int in  = mlp->input_size;
    int hid = mlp->hidden_size;
    int out = mlp->output_size;

    /* Copy training data to device */
    float *d_inputs;
    int   *d_targets;
    CUDA_CHECK(cudaMalloc(&d_inputs,  (size_t)num_samples * in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, (size_t)num_samples * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_inputs,  inputs,  (size_t)num_samples * in * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets, (size_t)num_samples * sizeof(int),        cudaMemcpyHostToDevice));

    /* Workspace buffers on device */
    float *d_hidden, *d_output, *d_d_output, *d_d_hidden;
    float *d_grad_W1, *d_grad_W2, *d_grad_b1, *d_grad_b2;
    float *d_losses, *d_loss_sum;

    CUDA_CHECK(cudaMalloc(&d_hidden,    (size_t)batch_size * hid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output,    (size_t)batch_size * out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_output,  (size_t)batch_size * out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_hidden,  (size_t)batch_size * hid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_W1,   (size_t)in  * hid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_W2,   (size_t)hid * out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_b1,   (size_t)hid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_b2,   (size_t)out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_losses,    (size_t)batch_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_loss_sum, sizeof(float)));

    int threads = 256;
    int num_batches = (num_samples + batch_size - 1) / batch_size;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        *d_loss_sum = 0.0f;

        for (int batch = 0; batch < num_batches; ++batch) {
            int start = batch * batch_size;
            int bs = batch_size;
            if (start + bs > num_samples) bs = num_samples - start;

            float *X = d_inputs  + start * in;
            int   *Y = d_targets + start;

            /* ---- Forward: hidden = ReLU(X @ W1 + b1) ---- */
            sgemm_nn(bs, hid, in, X, mlp->input_weights, d_hidden);
            int n_hid = bs * hid;
            bias_relu_kernel<<<(n_hid + threads - 1) / threads, threads>>>(
                d_hidden, mlp->hidden_biases, n_hid, hid);

            /* ---- Forward: output = softmax(hidden @ W2 + b2) ---- */
            sgemm_nn(bs, out, hid, d_hidden, mlp->output_weights, d_output);
            bias_softmax_kernel<<<(bs + threads - 1) / threads, threads>>>(
                d_output, mlp->output_biases, bs, out);

            /* ---- Loss + d_output gradient ---- */
            ce_grad_kernel<<<(bs + threads - 1) / threads, threads>>>(
                d_output, Y, d_d_output, d_losses, bs, out);

            /* Accumulate batch loss */
            reduce_sum_kernel<<<(bs + threads - 1) / threads, threads, threads * sizeof(float)>>>(
                d_losses, d_loss_sum, bs);

            /* ---- Backward: GEMM for gradients ---- */

            /* grad_W2 = hidden^T @ d_output */
            sgemm_tn(hid, out, bs, d_hidden, d_d_output, d_grad_W2);

            /* grad_b2 = colsum(d_output) */
            col_sum_kernel<<<(out + threads - 1) / threads, threads>>>(
                d_d_output, d_grad_b2, bs, out);

            /* d_hidden = d_output @ W2^T */
            sgemm_nt(bs, hid, out, d_d_output, mlp->output_weights, d_d_hidden);

            /* Apply ReLU mask */
            relu_mask_kernel<<<(n_hid + threads - 1) / threads, threads>>>(
                d_d_hidden, d_hidden, n_hid);

            /* grad_W1 = X^T @ d_hidden */
            sgemm_tn(in, hid, bs, X, d_d_hidden, d_grad_W1);

            /* grad_b1 = colsum(d_hidden) */
            col_sum_kernel<<<(hid + threads - 1) / threads, threads>>>(
                d_d_hidden, d_grad_b1, bs, hid);

            /* ---- SGD update ---- */
            float lr_s = learning_rate / (float)bs;
            int n_iw = in * hid;
            int n_ow = hid * out;
            sgd_kernel<<<(n_iw + threads - 1) / threads, threads>>>(
                mlp->input_weights, d_grad_W1, lr_s, n_iw);
            sgd_kernel<<<(n_ow + threads - 1) / threads, threads>>>(
                mlp->output_weights, d_grad_W2, lr_s, n_ow);
            sgd_kernel<<<(hid + threads - 1) / threads, threads>>>(
                mlp->hidden_biases, d_grad_b1, lr_s, hid);
            sgd_kernel<<<(out + threads - 1) / threads, threads>>>(
                mlp->output_biases, d_grad_b2, lr_s, out);
        }

        if (epoch % 100 == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("  Epoch %4d  loss: %.4f\n", epoch, *d_loss_sum / num_samples);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_inputs);   cudaFree(d_targets);
    cudaFree(d_hidden);   cudaFree(d_output);
    cudaFree(d_d_output); cudaFree(d_d_hidden);
    cudaFree(d_grad_W1);  cudaFree(d_grad_W2);
    cudaFree(d_grad_b1);  cudaFree(d_grad_b2);
    cudaFree(d_losses);   cudaFree(d_loss_sum);
}

/* ------------------------------------------------------------------ */
/*  Evaluation: cuBLAS batched forward                               */
/* ------------------------------------------------------------------ */

/* Evaluate kernel: compute loss + argmax per sample */
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

void mlp_evaluate(MLP *mlp, float *inputs, int *targets, int num_samples,
                  float *loss, float *accuracy) {
    int in  = mlp->input_size;
    int hid = mlp->hidden_size;
    int out = mlp->output_size;

    float *d_inputs;
    int   *d_targets;
    CUDA_CHECK(cudaMalloc(&d_inputs,  (size_t)num_samples * in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, (size_t)num_samples * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_inputs,  inputs,  (size_t)num_samples * in * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets, (size_t)num_samples * sizeof(int),        cudaMemcpyHostToDevice));

    float *d_hidden, *d_output, *d_losses;
    float *d_loss_sum;
    int   *d_correct;
    CUDA_CHECK(cudaMalloc(&d_hidden, (size_t)num_samples * hid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, (size_t)num_samples * out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_losses, (size_t)num_samples * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_loss_sum, sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_correct,  sizeof(int)));

    *d_loss_sum = 0.0f;
    *d_correct  = 0;

    int threads = 256;

    /* Forward: hidden = ReLU(X @ W1 + b1) */
    sgemm_nn(num_samples, hid, in, d_inputs, mlp->input_weights, d_hidden);
    int n_hid = num_samples * hid;
    bias_relu_kernel<<<(n_hid + threads - 1) / threads, threads>>>(
        d_hidden, mlp->hidden_biases, n_hid, hid);

    /* Forward: output = softmax(hidden @ W2 + b2) */
    sgemm_nn(num_samples, out, hid, d_hidden, mlp->output_weights, d_output);
    bias_softmax_kernel<<<(num_samples + threads - 1) / threads, threads>>>(
        d_output, mlp->output_biases, num_samples, out);

    /* Metrics */
    eval_metrics_kernel<<<(num_samples + threads - 1) / threads, threads>>>(
        d_output, d_targets, d_losses, d_correct, num_samples, out);

    reduce_sum_kernel<<<(num_samples + threads - 1) / threads, threads,
                        threads * sizeof(float)>>>(
        d_losses, d_loss_sum, num_samples);

    CUDA_CHECK(cudaDeviceSynchronize());

    *loss = *d_loss_sum / num_samples;
    *accuracy = (float)(*d_correct) / num_samples * 100.0f;

    cudaFree(d_inputs);   cudaFree(d_targets);
    cudaFree(d_hidden);   cudaFree(d_output);
    cudaFree(d_losses);   cudaFree(d_loss_sum);
    cudaFree(d_correct);
}
