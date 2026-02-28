#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Tiled Matrix Multiplication Kernels (NO cuBLAS)                   */
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
/*  CNN-specific kernels: im2col, col2im, avg_pool, add_bias          */
/* ------------------------------------------------------------------ */

/* ---- Per-sample kernels (kept for interface compatibility) ---- */

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

extern "C" void launch_im2col(const float *input, float *col,
                                int C, int H, int W,
                                int kH, int kW, int stride,
                                int OH, int OW) {
    int total = C * kH * kW * OH * OW;
    int threads = 256;
    im2col_kernel<<<(total + threads - 1) / threads, threads>>>(
        input, col, C, H, W, kH, kW, stride, OH, OW);
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

extern "C" void launch_col2im(const float *col, float *output,
                                int C, int H, int W,
                                int kH, int kW, int stride,
                                int OH, int OW) {
    int total = C * kH * kW * OH * OW;
    int threads = 256;
    col2im_kernel<<<(total + threads - 1) / threads, threads>>>(
        col, output, C, H, W, kH, kW, stride, OH, OW);
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

extern "C" void launch_avg_pool_forward(const float *input, float *output,
                                          int C, int H, int W,
                                          int pool_size, int OH, int OW) {
    int total = C * OH * OW;
    int threads = 256;
    avg_pool_forward_kernel<<<(total + threads - 1) / threads, threads>>>(
        input, output, C, H, W, pool_size, OH, OW);
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

extern "C" void launch_avg_pool_backward(const float *grad_output, float *grad_input,
                                           int C, int H, int W,
                                           int pool_size, int OH, int OW) {
    int total = C * H * W;
    int threads = 256;
    avg_pool_backward_kernel<<<(total + threads - 1) / threads, threads>>>(
        grad_output, grad_input, C, H, W, pool_size, OH, OW);
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

    col[idx] = input[c * B * H * W + b * H * W + h * W + w];
}

extern "C" void launch_im2col_batch(const float *input, float *col,
                                      int C, int B, int H, int W,
                                      int kH, int kW, int stride,
                                      int OH, int OW) {
    int total = C * kH * kW * B * OH * OW;
    int threads = 256;
    im2col_batch_kernel<<<(total + threads - 1) / threads, threads>>>(
        input, col, C, B, H, W, kH, kW, stride, OH, OW);
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

extern "C" void launch_col2im_batch(const float *col, float *output,
                                      int C, int B, int H, int W,
                                      int kH, int kW, int stride,
                                      int OH, int OW) {
    int total = C * kH * kW * B * OH * OW;
    int threads = 256;
    col2im_batch_kernel<<<(total + threads - 1) / threads, threads>>>(
        col, output, C, B, H, W, kH, kW, stride, OH, OW);
}

/*
 * Batched avg pool forward: input [C, B*H*W] -> output [C, B*PH*PW]
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

extern "C" void launch_avg_pool_forward_batch(const float *input, float *output,
                                                int C, int B, int H, int W,
                                                int pool_size, int PH, int PW) {
    int total = C * B * PH * PW;
    int threads = 256;
    avg_pool_forward_batch_kernel<<<(total + threads - 1) / threads, threads>>>(
        input, output, C, B, H, W, pool_size, PH, PW);
}

/*
 * Batched avg pool backward: grad_output [C, B*PH*PW] -> grad_input [C, B*H*W]
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

extern "C" void launch_avg_pool_backward_batch(const float *grad_output, float *grad_input,
                                                 int C, int B, int H, int W,
                                                 int pool_size, int PH, int PW) {
    int total = C * B * H * W;
    int threads = 256;
    avg_pool_backward_batch_kernel<<<(total + threads - 1) / threads, threads>>>(
        grad_output, grad_input, C, B, H, W, pool_size, PH, PW);
}

/*
 * Transpose [C, B*S] -> [B, C*S]  (channel-major to batch-major)
 */
__global__ void transpose_cb_to_bc_kernel(const float *src, float *dst,
                                            int C, int B, int S) {
    int total = C * B * S;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int s = idx % S;
    int tmp = idx / S;
    int b = tmp % B;
    int c = tmp / B;

    dst[b * C * S + c * S + s] = src[idx];
}

extern "C" void launch_transpose_cb_to_bc(const float *src, float *dst,
                                            int C, int B, int S) {
    int total = C * B * S;
    int threads = 256;
    transpose_cb_to_bc_kernel<<<(total + threads - 1) / threads, threads>>>(src, dst, C, B, S);
}

/*
 * Transpose [B, C*S] -> [C, B*S]  (batch-major to channel-major)
 */
__global__ void transpose_bc_to_cb_kernel(const float *src, float *dst,
                                            int B, int C, int S) {
    int total = B * C * S;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int s = idx % S;
    int tmp = idx / S;
    int c = tmp % C;
    int b = tmp / C;

    dst[c * B * S + b * S + s] = src[idx];
}

extern "C" void launch_transpose_bc_to_cb(const float *src, float *dst,
                                            int B, int C, int S) {
    int total = B * C * S;
    int threads = 256;
    transpose_bc_to_cb_kernel<<<(total + threads - 1) / threads, threads>>>(src, dst, B, C, S);
}

/* Add bias to [channels, spatial] matrix */
__global__ void add_bias_kernel(float *x, const float *bias, int channels, int spatial) {
    int total = channels * spatial;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int c = idx / spatial;
        x[idx] += bias[c];
    }
}

extern "C" void launch_add_bias(float *x, const float *bias, int channels, int spatial) {
    int total = channels * spatial;
    int threads = 256;
    add_bias_kernel<<<(total + threads - 1) / threads, threads>>>(x, bias, channels, spatial);
}

/* Bias gradient for conv: sum spatial dims per channel */
__global__ void bias_grad_kernel(const float *grad, float *grad_bias, int channels, int spatial) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float sum = 0.0f;
    const float *gc = grad + c * spatial;
    for (int j = 0; j < spatial; j++)
        sum += gc[j];
    grad_bias[c] = sum;
}

extern "C" void launch_bias_grad(const float *grad, float *grad_bias, int channels, int spatial) {
    int threads = 256;
    bias_grad_kernel<<<(channels + threads - 1) / threads, threads>>>(grad, grad_bias, channels, spatial);
}

/* ------------------------------------------------------------------ */
/*  Shared elementwise kernels                                        */
/* ------------------------------------------------------------------ */

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

__global__ void relu_forward_kernel(float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        if (x[idx] < 0.0f) x[idx] = 0.0f;
}

extern "C" void launch_relu_forward(float *x, int n) {
    int threads = 256;
    relu_forward_kernel<<<(n + threads - 1) / threads, threads>>>(x, n);
}

__global__ void bias_softmax_kernel(float *output, const float *bias, int bs, int out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= bs) return;
    float *o = output + row * out;
    float mx = -1e30f;
    for (int j = 0; j < out; ++j) {
        o[j] += bias[j];
        if (o[j] > mx) mx = o[j];
    }
    float s = 0.0f;
    for (int j = 0; j < out; ++j) {
        o[j] = expf(o[j] - mx);
        s += o[j];
    }
    for (int j = 0; j < out; ++j) o[j] /= s;
}

extern "C" void launch_bias_softmax(float *output, const float *bias, int bs, int out) {
    int threads = 256;
    bias_softmax_kernel<<<(bs + threads - 1) / threads, threads>>>(output, bias, bs, out);
}

__global__ void ce_grad_kernel(const float *output, const int *targets,
                               float *d_output, float *losses, int bs, int out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= bs) return;
    const float *o = output + row * out;
    float *d = d_output + row * out;
    int t = targets[row];
    losses[row] = -logf(o[t] + 1e-7f);
    for (int j = 0; j < out; ++j)
        d[j] = o[j] - ((t == j) ? 1.0f : 0.0f);
}

extern "C" void launch_ce_grad(const float *output, const int *targets,
                                float *d_output, float *losses, int bs, int out) {
    int threads = 256;
    ce_grad_kernel<<<(bs + threads - 1) / threads, threads>>>(output, targets, d_output, losses, bs, out);
}

__global__ void relu_mask_kernel(float *d_hidden, const float *hidden, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
        if (hidden[idx] <= 0.0f) d_hidden[idx] = 0.0f;
}

extern "C" void launch_relu_mask(float *d_hidden, const float *hidden, int total) {
    int threads = 256;
    relu_mask_kernel<<<(total + threads - 1) / threads, threads>>>(d_hidden, hidden, total);
}

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

__global__ void sgd_kernel(float *param, const float *grad, float lr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) param[idx] -= lr * grad[idx];
}

extern "C" void launch_sgd(float *param, const float *grad, float lr, int n) {
    int threads = 256;
    sgd_kernel<<<(n + threads - 1) / threads, threads>>>(param, grad, lr, n);
}

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
    reduce_sum_kernel<<<(n + threads - 1) / threads, threads, threads * sizeof(float)>>>(arr, result, n);
}

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

__global__ void eval_metrics_kernel(const float *output, const int *targets,
                                    float *losses, int *correct, int bs, int out) {
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
    eval_metrics_kernel<<<(bs + threads - 1) / threads, threads>>>(output, targets, losses, correct, bs, out);
}
