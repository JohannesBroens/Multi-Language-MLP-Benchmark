#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  CNN-specific kernels: NCHW direct convolution + pool               */
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

/* ------------------------------------------------------------------ */
/*  NCHW batched kernels — direct convolution (no im2col)             */
/* ------------------------------------------------------------------ */

/*
 * Fused conv + bias + ReLU.  NCHW layout.
 * input  [B, C_in, H, W]
 * weight [C_out, C_in*K*K]
 * bias   [C_out]
 * output [B, C_out, OH, OW]
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

extern "C" void launch_conv_fwd_bias_relu_nchw(
        const float *input, const float *weight, const float *bias,
        float *output,
        int B, int C_in, int H, int W, int C_out, int K, int OH, int OW) {
    int spatial = B * OH * OW;
    int threads = 256;
    int CKK = C_in * K * K;
    dim3 grid((spatial + threads - 1) / threads, C_out);
    conv_fwd_bias_relu_nchw<<<grid, threads, CKK * sizeof(float)>>>(
        input, weight, bias, output, B, C_in, H, W, C_out, K, OH, OW);
}

/*
 * Avg pool forward NCHW.  [B, C, H, W] -> [B, C, PH, PW].
 */
__global__ void avg_pool_fwd_nchw_kernel(
        const float *input, float *output,
        int B, int C, int H, int W, int pool, int PH, int PW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * PH * PW;
    if (idx >= total) return;

    int pw_idx = idx % PW;
    int tmp = idx / PW;
    int ph = tmp % PH;
    int tmp2 = tmp / PH;
    int c = tmp2 % C;
    int b = tmp2 / C;

    float scale = 1.0f / (pool * pool);
    float sum = 0.0f;
    const float *in_bc = input + b * C * H * W + c * H * W;
    int h0 = ph * pool, w0 = pw_idx * pool;
    for (int i = 0; i < pool; i++)
        for (int j = 0; j < pool; j++)
            sum += in_bc[(h0 + i) * W + w0 + j];
    output[idx] = sum * scale;
}

extern "C" void launch_avg_pool_fwd_nchw(
        const float *input, float *output,
        int B, int C, int H, int W, int pool, int PH, int PW) {
    int total = B * C * PH * PW;
    int threads = 256;
    avg_pool_fwd_nchw_kernel<<<(total + threads - 1) / threads, threads>>>(
        input, output, B, C, H, W, pool, PH, PW);
}

/*
 * Avg pool backward NCHW.
 * grad_output [B, C, PH, PW] -> grad_input [B, C, H, W].
 */
__global__ void avg_pool_bwd_nchw_kernel(
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

extern "C" void launch_avg_pool_bwd_nchw(
        const float *grad_output, float *grad_input,
        int B, int C, int H, int W, int pool, int PH, int PW) {
    int total = B * C * H * W;
    int threads = 256;
    avg_pool_bwd_nchw_kernel<<<(total + threads - 1) / threads, threads>>>(
        grad_output, grad_input, B, C, H, W, pool, PH, PW);
}

/*
 * Direct conv input gradient NCHW.  No col2im, no atomicAdd.
 * d_output [B, C_out, OH, OW]
 * weight   [C_out, C_in*K*K]
 * d_input  [B, C_in, H, W]
 */
__global__ void conv_input_grad_nchw_kernel(
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
            int o_h = ih - kh;
            if (o_h < 0 || o_h >= OH) continue;
            for (int kw = 0; kw < K; kw++) {
                int o_w = iw - kw;
                if (o_w < 0 || o_w >= OW) continue;
                sum += d_bco[o_h * OW + o_w] * w_coci[kh * K + kw];
            }
        }
    }
    d_input[idx] = sum;
}

extern "C" void launch_conv_input_grad_nchw(
        const float *d_output, const float *weight, float *d_input,
        int B, int C_out, int C_in, int H, int W, int K, int OH, int OW) {
    int total = B * C_in * H * W;
    int threads = 256;
    conv_input_grad_nchw_kernel<<<(total + threads - 1) / threads, threads>>>(
        d_output, weight, d_input, B, C_out, C_in, H, W, K, OH, OW);
}

/*
 * Conv weight gradient via parallel reduction.  NCHW.
 * One block per weight element.  Threads reduce over B*OH*OW.
 */
__global__ void conv_weight_grad_nchw_kernel(
        const float *d_output, const float *input, float *grad_w,
        int B, int C_out, int C_in, int H, int W, int K, int OH, int OW) {
    extern __shared__ float sdata[];

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
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) grad_w[w_idx] = sdata[0];
}

extern "C" void launch_conv_weight_grad_nchw(
        const float *d_output, const float *input, float *grad_w,
        int num_weights, int B, int C_out, int C_in, int H, int W, int K, int OH, int OW) {
    int threads = 256;
    conv_weight_grad_nchw_kernel<<<num_weights, threads, threads * sizeof(float)>>>(
        d_output, input, grad_w, B, C_out, C_in, H, W, K, OH, OW);
}

/*
 * Conv bias gradient via parallel reduction.  NCHW.
 * One block per output channel.
 */
__global__ void conv_bias_grad_nchw_kernel(
        const float *d_output, float *grad_bias,
        int B, int C_out, int OH, int OW) {
    extern __shared__ float sdata[];

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
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) grad_bias[co] = sdata[0];
}

extern "C" void launch_conv_bias_grad_nchw(
        const float *d_output, float *grad_bias,
        int C_out, int B, int OH, int OW) {
    int threads = 256;
    conv_bias_grad_nchw_kernel<<<C_out, threads, threads * sizeof(float)>>>(
        d_output, grad_bias, B, C_out, OH, OW);
}

/* ------------------------------------------------------------------ */
/*  Optimized backward pass kernels                                    */
/* ------------------------------------------------------------------ */

/*
 * Batched im2col for NCHW input. Extracts patches for all samples.
 * input [B, C_in, H, W] -> col [C_in*K*K, B*OH*OW]
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

extern "C" void launch_im2col_nchw_batch(
        const float *input, float *col,
        int B, int C_in, int H, int W, int K, int OH, int OW) {
    int total = C_in * K * K * B * OH * OW;
    int threads = 256;
    im2col_nchw_batch_kernel<<<(total + threads - 1) / threads, threads>>>(
        input, col, B, C_in, H, W, K, OH, OW);
}

/*
 * Fused NCHW->CM transpose + bias gradient.
 * Transposes [B, C, H, W] -> [C, B*H*W] while accumulating per-channel sums.
 * bias_grad must be zeroed before launch (uses atomicAdd on global).
 */
__global__ void nchw_to_cm_bias_grad_kernel(
        const float *input, float *output, float *bias_grad,
        int B, int C, int H, int W) {
    extern __shared__ float s_bias[];

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

    for (int i = threadIdx.x; i < C; i += blockDim.x)
        atomicAdd(&bias_grad[i], s_bias[i]);
}

extern "C" void launch_nchw_to_cm_bias_grad(
        const float *input, float *output, float *bias_grad,
        int B, int C, int H, int W) {
    int total = B * C * H * W;
    int threads = 256;
    nchw_to_cm_bias_grad_kernel<<<(total + threads - 1) / threads, threads, C * sizeof(float)>>>(
        input, output, bias_grad, B, C, H, W);
}

/*
 * Gather-based col2im for NCHW output. No atomicAdd.
 * col [C_in*K*K, B*OH*OW] -> output [B, C_in, H, W]
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

extern "C" void launch_col2im_gather_nchw(
        const float *col, float *output,
        int B, int C_in, int H, int W, int K, int OH, int OW) {
    int total = B * C_in * H * W;
    int threads = 256;
    col2im_gather_nchw_kernel<<<(total + threads - 1) / threads, threads>>>(
        col, output, B, C_in, H, W, K, OH, OW);
}

/* ------------------------------------------------------------------ */
/*  Shared elementwise kernels (same as MLP crates)                   */
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
