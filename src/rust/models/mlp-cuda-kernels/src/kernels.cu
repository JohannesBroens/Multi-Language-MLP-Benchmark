#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Tiled Matrix Multiplication Kernels                               */
/* ------------------------------------------------------------------ */

#define TILE_DIM 16

// C[M,N] = A[M,K] @ B[K,N]  — shared memory tiled
__global__ void matmul_nn_kernel(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        int ak = t * TILE_DIM + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;

        int bk = t * TILE_DIM + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_DIM; i++)
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

extern "C" void launch_matmul_nn(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    dim3 block(TILE_DIM, TILE_DIM);
    matmul_nn_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

// C[M,N] = A^T[M,K] @ B[K,N],  A stored as [K,M]
__global__ void matmul_tn_kernel(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        // A is stored as [K,M], transposed access: A^T[row, ak] = A[ak, row] = A[ak * M + row]
        int ak = t * TILE_DIM + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[ak * M + row] : 0.0f;

        int bk = t * TILE_DIM + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_DIM; i++)
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

extern "C" void launch_matmul_tn(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    dim3 block(TILE_DIM, TILE_DIM);
    matmul_tn_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

// C[M,N] = A[M,K] @ B^T[K,N],  B stored as [N,K]
__global__ void matmul_nt_kernel(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        int ak = t * TILE_DIM + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;

        // B is stored as [N,K], transposed access: B^T[bk, col] = B[col, bk] = B[col * K + bk]
        int bk = t * TILE_DIM + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[col * K + bk] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_DIM; i++)
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

extern "C" void launch_matmul_nt(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    dim3 block(TILE_DIM, TILE_DIM);
    matmul_nt_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

/* ------------------------------------------------------------------ */
/*  Element-wise CUDA kernels                                         */
/* ------------------------------------------------------------------ */

/* Add bias vector then apply ReLU: hidden[i] = max(0, hidden[i] + bias[i % hid]) */
__global__ void bias_relu_kernel(float *hidden, const float *bias, int total, int hid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float v = hidden[idx] + bias[idx % hid];
        hidden[idx] = (v > 0.0f) ? v : 0.0f;
    }
}

extern "C" void launch_bias_relu(float *hidden, const float *bias, int total, int hid) {
    int threads = 256;
    bias_relu_kernel<<<(total + threads - 1) / threads, threads>>>(hidden, bias, total, hid);
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

extern "C" void launch_bias_softmax(float *output, const float *bias, int bs, int out) {
    int threads = 256;
    bias_softmax_kernel<<<(bs + threads - 1) / threads, threads>>>(output, bias, bs, out);
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

extern "C" void launch_ce_grad(const float *output, const int *targets,
                                float *d_output, float *losses, int bs, int out) {
    int threads = 256;
    ce_grad_kernel<<<(bs + threads - 1) / threads, threads>>>(
        output, targets, d_output, losses, bs, out);
}

/* Apply ReLU mask: d_hidden[i] = (hidden[i] > 0) ? d_hidden[i] : 0 */
__global__ void relu_mask_kernel(float *d_hidden, const float *hidden, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
        if (hidden[idx] <= 0.0f) d_hidden[idx] = 0.0f;
}

extern "C" void launch_relu_mask(float *d_hidden, const float *hidden, int total) {
    int threads = 256;
    relu_mask_kernel<<<(total + threads - 1) / threads, threads>>>(d_hidden, hidden, total);
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

extern "C" void launch_col_sum(const float *mat, float *out_vec, int rows, int cols) {
    int threads = 256;
    col_sum_kernel<<<(cols + threads - 1) / threads, threads>>>(mat, out_vec, rows, cols);
}

/* SGD update: param[i] -= lr * grad[i] */
__global__ void sgd_kernel(float *param, const float *grad, float lr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        param[idx] -= lr * grad[idx];
}

extern "C" void launch_sgd(float *param, const float *grad, float lr, int n) {
    int threads = 256;
    sgd_kernel<<<(n + threads - 1) / threads, threads>>>(param, grad, lr, n);
}

/* Reduce sum with shared memory */
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

extern "C" void launch_reduce_sum(const float *arr, float *result, int n) {
    int threads = 256;
    reduce_sum_kernel<<<(n + threads - 1) / threads, threads, threads * sizeof(float)>>>(
        arr, result, n);
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

extern "C" void launch_init_weights(float *weights, int size, float scale, unsigned long seed) {
    int threads = 256;
    init_weights_kernel<<<(size + threads - 1) / threads, threads>>>(weights, size, scale, seed);
}

/* ------------------------------------------------------------------ */
/*  Eval metrics: loss + argmax accuracy                              */
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

extern "C" void launch_eval_metrics(const float *output, const int *targets,
                                     float *losses, int *correct, int bs, int out) {
    int threads = 256;
    eval_metrics_kernel<<<(bs + threads - 1) / threads, threads>>>(
        output, targets, losses, correct, bs, out);
}
