#include "mlp.h"
#include "nn_ops.h"
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

void mlp_initialize(MLP *mlp, int input_size, int hidden_size, int output_size,
                    int num_hidden_layers) {
    mlp->input_size = input_size;
    mlp->hidden_size = hidden_size;
    mlp->output_size = output_size;
    mlp->num_layers = num_hidden_layers + 1;

    int nl = mlp->num_layers;

    /* Build layer_sizes on host */
    mlp->layer_sizes = (int *)malloc((nl + 1) * sizeof(int));
    mlp->layer_sizes[0] = input_size;
    for (int i = 1; i <= num_hidden_layers; ++i)
        mlp->layer_sizes[i] = hidden_size;
    mlp->layer_sizes[nl] = output_size;

    /* Allocate weight and bias pointer arrays */
    mlp->weights = (float **)malloc(nl * sizeof(float *));
    mlp->biases  = (float **)malloc(nl * sizeof(float *));

    int threads = 256;
    unsigned long seed = (unsigned long)time(NULL);

    for (int i = 0; i < nl; ++i) {
        int fan_in  = mlp->layer_sizes[i];
        int fan_out = mlp->layer_sizes[i + 1];

        CUDA_CHECK(cudaMallocManaged(&mlp->weights[i], fan_in * fan_out * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&mlp->biases[i],  fan_out * sizeof(float)));
        cudaMemset(mlp->biases[i], 0, fan_out * sizeof(float));

        float scale = sqrtf(2.0f / (float)fan_in);
        int blocks = (fan_in * fan_out + threads - 1) / threads;
        init_weights_kernel<<<blocks, threads>>>(mlp->weights[i],
                                                  fan_in * fan_out, scale, seed + i);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    /* Create cuBLAS handle */
    if (!cublas_handle)
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
}

void mlp_free(MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; ++i) {
        cudaFree(mlp->weights[i]);
        cudaFree(mlp->biases[i]);
    }
    free(mlp->weights);
    free(mlp->biases);
    free(mlp->layer_sizes);

    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = NULL;
    }
}

/* ------------------------------------------------------------------ */
/*  Training: cuBLAS GEMM + elementwise kernels                      */
/* ------------------------------------------------------------------ */

void mlp_train(MLP *mlp, float *inputs, int *targets, int num_samples,
               int batch_size, int num_epochs, float learning_rate,
               OptimizerType optimizer, SchedulerType scheduler) {
    int nl = mlp->num_layers;
    int num_hidden = nl - 1;
    int hid = mlp->hidden_size;
    int out = mlp->output_size;
    int in  = mlp->input_size;

    /* Copy training data to device */
    float *d_inputs;
    int   *d_targets;
    CUDA_CHECK(cudaMalloc(&d_inputs,  (size_t)num_samples * in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, (size_t)num_samples * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_inputs,  inputs,  (size_t)num_samples * in * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets, (size_t)num_samples * sizeof(int),        cudaMemcpyHostToDevice));

    /* Hidden activation buffers (needed for backward) */
    float **d_acts = (float **)malloc(num_hidden * sizeof(float *));
    for (int i = 0; i < num_hidden; ++i)
        CUDA_CHECK(cudaMalloc(&d_acts[i], (size_t)batch_size * hid * sizeof(float)));

    /* Output + gradient buffers */
    float *d_output, *d_d_output;
    CUDA_CHECK(cudaMalloc(&d_output,   (size_t)batch_size * out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_output, (size_t)batch_size * out * sizeof(float)));

    /* Gradient propagation buffers (ping-pong) */
    int max_dim = hid > out ? hid : out;
    float *d_grad_cur, *d_grad_prev;
    CUDA_CHECK(cudaMalloc(&d_grad_cur,  (size_t)batch_size * max_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_prev, (size_t)batch_size * max_dim * sizeof(float)));

    /* Per-layer weight/bias gradient buffers */
    float **d_grad_W = (float **)malloc(nl * sizeof(float *));
    float **d_grad_b = (float **)malloc(nl * sizeof(float *));
    for (int i = 0; i < nl; ++i) {
        int fan_in  = mlp->layer_sizes[i];
        int fan_out = mlp->layer_sizes[i + 1];
        CUDA_CHECK(cudaMalloc(&d_grad_W[i], (size_t)fan_in * fan_out * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_b[i], (size_t)fan_out * sizeof(float)));
    }

    float *d_losses, *d_loss_sum;
    CUDA_CHECK(cudaMalloc(&d_losses, (size_t)batch_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_loss_sum, sizeof(float)));

    /* Adam optimizer state */
    AdamState *adam_W = NULL, *adam_b = NULL;
    int step = 0;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    if (optimizer == OPT_ADAM) {
        adam_W = (AdamState *)calloc(nl, sizeof(AdamState));
        adam_b = (AdamState *)calloc(nl, sizeof(AdamState));
        for (int i = 0; i < nl; ++i) {
            int fan_in  = mlp->layer_sizes[i];
            int fan_out = mlp->layer_sizes[i + 1];
            nn_adam_state_init(&adam_W[i], fan_in * fan_out);
            nn_adam_state_init(&adam_b[i], fan_out);
        }
    }

    int threads = 256;
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    int warmup_epochs = (scheduler == SCHED_COSINE) ? (int)(num_epochs * 0.05f) : 0;
    if (warmup_epochs < 1 && scheduler == SCHED_COSINE) warmup_epochs = 1;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float lr = (scheduler == SCHED_COSINE)
            ? nn_cosine_lr(epoch, num_epochs, learning_rate, warmup_epochs, 1e-6f)
            : learning_rate;
        *d_loss_sum = 0.0f;

        for (int batch = 0; batch < num_batches; ++batch) {
            int start = batch * batch_size;
            int bs = batch_size;
            if (start + bs > num_samples) bs = num_samples - start;

            float *X = d_inputs  + start * in;
            int   *Y = d_targets + start;

            /* ---- Forward ---- */
            for (int i = 0; i < nl; ++i) {
                int in_dim  = mlp->layer_sizes[i];
                int out_dim = mlp->layer_sizes[i + 1];
                const float *input_act = (i == 0) ? X : d_acts[i - 1];
                float *out_act = (i < num_hidden) ? d_acts[i] : d_output;

                sgemm_nn(bs, out_dim, in_dim, input_act, mlp->weights[i], out_act);

                if (i < num_hidden) {
                    int n_elem = bs * out_dim;
                    bias_relu_kernel<<<(n_elem + threads - 1) / threads, threads>>>(
                        out_act, mlp->biases[i], n_elem, out_dim);
                } else {
                    bias_softmax_kernel<<<(bs + threads - 1) / threads, threads>>>(
                        out_act, mlp->biases[i], bs, out_dim);
                }
            }

            /* ---- Loss + d_output gradient ---- */
            ce_grad_kernel<<<(bs + threads - 1) / threads, threads>>>(
                d_output, Y, d_d_output, d_losses, bs, out);

            reduce_sum_kernel<<<(bs + threads - 1) / threads, threads, threads * sizeof(float)>>>(
                d_losses, d_loss_sum, bs);

            /* ---- Backward ---- */
            float *dcur = d_d_output;

            for (int layer = nl - 1; layer >= 0; --layer) {
                int fan_in  = mlp->layer_sizes[layer];
                int fan_out = mlp->layer_sizes[layer + 1];
                const float *input_act = (layer == 0) ? X : d_acts[layer - 1];

                /* grad_W[layer] = input_act^T @ dcur */
                sgemm_tn(fan_in, fan_out, bs, input_act, dcur, d_grad_W[layer]);

                /* grad_b[layer] = colsum(dcur) */
                col_sum_kernel<<<(fan_out + threads - 1) / threads, threads>>>(
                    dcur, d_grad_b[layer], bs, fan_out);

                /* Propagate gradient to previous layer */
                if (layer > 0) {
                    float *dnext = (dcur == d_d_output) ? d_grad_cur :
                                   (dcur == d_grad_cur) ? d_grad_prev : d_grad_cur;
                    sgemm_nt(bs, fan_in, fan_out, dcur, mlp->weights[layer], dnext);

                    int n_elem = bs * fan_in;
                    relu_mask_kernel<<<(n_elem + threads - 1) / threads, threads>>>(
                        dnext, d_acts[layer - 1], n_elem);
                    dcur = dnext;
                }
            }

            /* ---- Parameter update ---- */
            float lr_s = lr / (float)bs;
            step++;

            for (int i = 0; i < nl; ++i) {
                int fan_in  = mlp->layer_sizes[i];
                int fan_out = mlp->layer_sizes[i + 1];
                int n_w = fan_in * fan_out;

                if (optimizer == OPT_ADAM) {
                    nn_adam_update(mlp->weights[i], d_grad_W[i], &adam_W[i], lr, beta1, beta2, eps, step);
                    nn_adam_update(mlp->biases[i], d_grad_b[i], &adam_b[i], lr, beta1, beta2, eps, step);
                } else {
                    sgd_kernel<<<(n_w + threads - 1) / threads, threads>>>(
                        mlp->weights[i], d_grad_W[i], lr_s, n_w);
                    sgd_kernel<<<(fan_out + threads - 1) / threads, threads>>>(
                        mlp->biases[i], d_grad_b[i], lr_s, fan_out);
                }
            }
        }

        if (epoch % 100 == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("  Epoch %4d  loss: %.4f\n", epoch, *d_loss_sum / num_samples);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (optimizer == OPT_ADAM) {
        for (int i = 0; i < nl; ++i) {
            nn_adam_state_free(&adam_W[i]);
            nn_adam_state_free(&adam_b[i]);
        }
        free(adam_W);
        free(adam_b);
    }

    cudaFree(d_inputs);   cudaFree(d_targets);
    for (int i = 0; i < num_hidden; ++i)
        cudaFree(d_acts[i]);
    free(d_acts);
    cudaFree(d_output);   cudaFree(d_d_output);
    cudaFree(d_grad_cur); cudaFree(d_grad_prev);
    for (int i = 0; i < nl; ++i) {
        cudaFree(d_grad_W[i]);
        cudaFree(d_grad_b[i]);
    }
    free(d_grad_W);       free(d_grad_b);
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
    int nl = mlp->num_layers;
    int num_hidden = nl - 1;
    int in  = mlp->input_size;
    int hid = mlp->hidden_size;
    int out = mlp->output_size;

    float *d_inputs;
    int   *d_targets;
    CUDA_CHECK(cudaMalloc(&d_inputs,  (size_t)num_samples * in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, (size_t)num_samples * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_inputs,  inputs,  (size_t)num_samples * in * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, targets, (size_t)num_samples * sizeof(int),        cudaMemcpyHostToDevice));

    /* Use ping-pong buffers for hidden activations (no backward needed) */
    float *d_buf0 = NULL, *d_buf1 = NULL;
    if (num_hidden > 0) {
        CUDA_CHECK(cudaMalloc(&d_buf0, (size_t)num_samples * hid * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_buf1, (size_t)num_samples * hid * sizeof(float)));
    }

    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, (size_t)num_samples * out * sizeof(float)));

    int threads = 256;

    /* Forward through all layers */
    for (int i = 0; i < nl; ++i) {
        int in_dim  = mlp->layer_sizes[i];
        int out_dim = mlp->layer_sizes[i + 1];

        const float *input_act;
        if (i == 0)
            input_act = d_inputs;
        else
            input_act = ((i - 1) % 2 == 0) ? d_buf0 : d_buf1;

        float *out_act;
        if (i < num_hidden)
            out_act = (i % 2 == 0) ? d_buf0 : d_buf1;
        else
            out_act = d_output;

        sgemm_nn(num_samples, out_dim, in_dim, input_act, mlp->weights[i], out_act);

        if (i < num_hidden) {
            int n_elem = num_samples * out_dim;
            bias_relu_kernel<<<(n_elem + threads - 1) / threads, threads>>>(
                out_act, mlp->biases[i], n_elem, out_dim);
        } else {
            bias_softmax_kernel<<<(num_samples + threads - 1) / threads, threads>>>(
                out_act, mlp->biases[i], num_samples, out_dim);
        }
    }

    /* Metrics */
    float *d_losses, *d_loss_sum;
    int   *d_correct;
    CUDA_CHECK(cudaMalloc(&d_losses, (size_t)num_samples * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_loss_sum, sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_correct,  sizeof(int)));

    *d_loss_sum = 0.0f;
    *d_correct  = 0;

    eval_metrics_kernel<<<(num_samples + threads - 1) / threads, threads>>>(
        d_output, d_targets, d_losses, d_correct, num_samples, out);

    reduce_sum_kernel<<<(num_samples + threads - 1) / threads, threads,
                        threads * sizeof(float)>>>(
        d_losses, d_loss_sum, num_samples);

    CUDA_CHECK(cudaDeviceSynchronize());

    *loss = *d_loss_sum / num_samples;
    *accuracy = (float)(*d_correct) / num_samples * 100.0f;

    cudaFree(d_inputs);   cudaFree(d_targets);
    if (d_buf0) cudaFree(d_buf0);
    if (d_buf1) cudaFree(d_buf1);
    cudaFree(d_output);
    cudaFree(d_losses);   cudaFree(d_loss_sum);
    cudaFree(d_correct);
}
