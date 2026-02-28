#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Element-wise CUDA kernels — copied from C mlp.cu                  */
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

/* Adam optimizer update */
__global__ void adam_kernel(float *param, const float *grad,
                            float *m, float *v,
                            float lr, float beta1, float beta2, float eps,
                            float bc1, float bc2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad[idx] * grad[idx];
        float m_hat = m[idx] / bc1;
        float v_hat = v[idx] / bc2;
        param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

extern "C" void launch_adam(float *param, const float *grad,
                             float *m, float *v,
                             float lr, float beta1, float beta2, float eps,
                             int t, int n) {
    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);
    int blocks = (n + 255) / 256;
    adam_kernel<<<blocks, 256>>>(param, grad, m, v, lr, beta1, beta2, eps, bc1, bc2, n);
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
