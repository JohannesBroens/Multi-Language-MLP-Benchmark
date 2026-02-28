#include "cnn.h"
#include "nn_ops.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  LeNet-5 layer dimensions                                          */
/* ------------------------------------------------------------------ */

/* Input: 1x28x28 */
#define IN_C  1
#define IN_H  28
#define IN_W  28

/* Conv1: 6 filters 5x5, stride 1, no padding -> 6x24x24 */
#define C1_OUT 6
#define C1_IN  1
#define C1_K   5
#define C1_OH  (IN_H - C1_K + 1)   /* 24 */
#define C1_OW  (IN_W - C1_K + 1)   /* 24 */
#define C1_COL_ROWS (C1_IN * C1_K * C1_K)  /* 25 */
#define C1_COL_COLS (C1_OH * C1_OW)         /* 576 */

/* Pool1: 2x2 avg -> 6x12x12 */
#define P1_SIZE 2
#define P1_H   (C1_OH / P1_SIZE)   /* 12 */
#define P1_W   (C1_OW / P1_SIZE)   /* 12 */

/* Conv2: 16 filters 5x5, stride 1, no padding -> 16x8x8 */
#define C2_OUT 16
#define C2_IN  6
#define C2_K   5
#define C2_OH  (P1_H - C2_K + 1)   /* 8 */
#define C2_OW  (P1_W - C2_K + 1)   /* 8 */
#define C2_COL_ROWS (C2_IN * C2_K * C2_K)  /* 150 */
#define C2_COL_COLS (C2_OH * C2_OW)         /* 64 */

/* Pool2: 2x2 avg -> 16x4x4 */
#define P2_SIZE 2
#define P2_H   (C2_OH / P2_SIZE)   /* 4 */
#define P2_W   (C2_OW / P2_SIZE)   /* 4 */

/* Flatten: 16*4*4 = 256 */
#define FLAT_SIZE (C2_OUT * P2_H * P2_W)  /* 256 */

/* FC layers */
#define FC1_OUT 120
#define FC2_OUT 84
#define NUM_CLASSES 10

/* ------------------------------------------------------------------ */
/*  Per-sample im2col / col2im / pooling  (cnn.h interface)           */
/* ------------------------------------------------------------------ */

void im2col(const float *input, int C, int H, int W,
            int kH, int kW, int stride, float *col) {
    int OH = (H - kH) / stride + 1;
    int OW = (W - kW) / stride + 1;
    int col_row = 0;

    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        int h = oh * stride + kh;
                        int w = ow * stride + kw;
                        col[col_row * (OH * OW) + oh * OW + ow] =
                            input[c * H * W + h * W + w];
                    }
                }
                col_row++;
            }
        }
    }
}

void col2im(const float *col, int C, int H, int W,
            int kH, int kW, int stride, float *output) {
    int OH = (H - kH) / stride + 1;
    int OW = (W - kW) / stride + 1;
    int col_row = 0;

    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        int h = oh * stride + kh;
                        int w = ow * stride + kw;
                        output[c * H * W + h * W + w] +=
                            col[col_row * (OH * OW) + oh * OW + ow];
                    }
                }
                col_row++;
            }
        }
    }
}

void avg_pool_forward(const float *input, float *output,
                      int C, int H, int W, int pool_size) {
    int OH = H / pool_size;
    int OW = W / pool_size;
    float scale = 1.0f / (pool_size * pool_size);

    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                float sum = 0.0f;
                for (int ph = 0; ph < pool_size; ph++) {
                    for (int pw = 0; pw < pool_size; pw++) {
                        int h = oh * pool_size + ph;
                        int w = ow * pool_size + pw;
                        sum += input[c * H * W + h * W + w];
                    }
                }
                output[c * OH * OW + oh * OW + ow] = sum * scale;
            }
        }
    }
}

void avg_pool_backward(const float *grad_output, float *grad_input,
                       int C, int H, int W, int pool_size) {
    int OH = H / pool_size;
    int OW = W / pool_size;
    float scale = 1.0f / (pool_size * pool_size);

    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                float g = grad_output[c * OH * OW + oh * OW + ow] * scale;
                for (int ph = 0; ph < pool_size; ph++) {
                    for (int pw = 0; pw < pool_size; pw++) {
                        int h = oh * pool_size + ph;
                        int w = ow * pool_size + pw;
                        grad_input[c * H * W + h * W + w] += g;
                    }
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Helper: add bias to [C_out, spatial] matrix (each row gets bias)  */
/* ------------------------------------------------------------------ */

static void add_bias(float *x, const float *bias, int channels, int spatial) {
    for (int c = 0; c < channels; c++) {
        float b = bias[c];
        float *xc = x + c * spatial;
        for (int j = 0; j < spatial; j++)
            xc[j] += b;
    }
}

/* ------------------------------------------------------------------ */
/*  Batched helpers: channel-major [C, B*spatial] layout               */
/*                                                                     */
/*  Data layout: for channel c, sample b, spatial position s:          */
/*    index = c * B * spatial + b * spatial + s                        */
/* ------------------------------------------------------------------ */

/*
 * Batched im2col: unfold patches for all B samples in one pass.
 * Input:  [C, B*H*W]  channel-major
 * Output: [K, B*OH*OW] where K = C*kH*kW
 */
static void im2col_batch(const float *input, int B, int C, int H, int W,
                          int kH, int kW, int stride, float *col) {
    int OH = (H - kH) / stride + 1;
    int OW = (W - kW) / stride + 1;
    int spatial_out = OH * OW;
    int col_cols = B * spatial_out;

    int col_row = 0;
    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                float *col_ptr = col + col_row * col_cols;
                for (int b = 0; b < B; b++) {
                    const float *in_b = input + c * (B * H * W) + b * (H * W);
                    float *out_b = col_ptr + b * spatial_out;
                    for (int oh = 0; oh < OH; oh++) {
                        for (int ow = 0; ow < OW; ow++) {
                            int h = oh * stride + kh;
                            int w = ow * stride + kw;
                            out_b[oh * OW + ow] = in_b[h * W + w];
                        }
                    }
                }
                col_row++;
            }
        }
    }
}

/*
 * Batched col2im: scatter gradients back for all B samples.
 * Input:  [K, B*OH*OW]
 * Output: [C, B*H*W]  (accumulates, caller must zero)
 */
static void col2im_batch(const float *col, int B, int C, int H, int W,
                          int kH, int kW, int stride, float *output) {
    int OH = (H - kH) / stride + 1;
    int OW = (W - kW) / stride + 1;
    int spatial_out = OH * OW;
    int col_cols = B * spatial_out;

    int col_row = 0;
    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                const float *col_ptr = col + col_row * col_cols;
                for (int b = 0; b < B; b++) {
                    float *out_b = output + c * (B * H * W) + b * (H * W);
                    const float *in_b = col_ptr + b * spatial_out;
                    for (int oh = 0; oh < OH; oh++) {
                        for (int ow = 0; ow < OW; ow++) {
                            int h = oh * stride + kh;
                            int w = ow * stride + kw;
                            out_b[h * W + w] += in_b[oh * OW + ow];
                        }
                    }
                }
                col_row++;
            }
        }
    }
}

/*
 * Batched average pooling forward.
 * Input:  [C, B*H*W]   Output: [C, B*PH*PW]
 */
static void avg_pool_forward_batch(const float *input, float *output,
                                    int C, int B, int H, int W, int pool_size) {
    int PH = H / pool_size;
    int PW = W / pool_size;
    float scale = 1.0f / (pool_size * pool_size);

    for (int c = 0; c < C; c++) {
        for (int b = 0; b < B; b++) {
            const float *in_cb = input + c * B * H * W + b * H * W;
            float *out_cb = output + c * B * PH * PW + b * PH * PW;
            for (int ph = 0; ph < PH; ph++) {
                for (int pw = 0; pw < PW; pw++) {
                    float sum = 0.0f;
                    for (int i = 0; i < pool_size; i++) {
                        for (int j = 0; j < pool_size; j++) {
                            sum += in_cb[(ph * pool_size + i) * W + (pw * pool_size + j)];
                        }
                    }
                    out_cb[ph * PW + pw] = sum * scale;
                }
            }
        }
    }
}

/*
 * Batched average pooling backward.
 * Input:  [C, B*PH*PW]   Output: [C, B*H*W]  (overwrites)
 */
static void avg_pool_backward_batch(const float *grad_output, float *grad_input,
                                     int C, int B, int H, int W, int pool_size) {
    int PH = H / pool_size;
    int PW = W / pool_size;
    float scale = 1.0f / (pool_size * pool_size);

    for (int c = 0; c < C; c++) {
        for (int b = 0; b < B; b++) {
            const float *go = grad_output + c * B * PH * PW + b * PH * PW;
            float *gi = grad_input + c * B * H * W + b * H * W;
            for (int ph = 0; ph < PH; ph++) {
                for (int pw = 0; pw < PW; pw++) {
                    float g = go[ph * PW + pw] * scale;
                    for (int i = 0; i < pool_size; i++) {
                        for (int j = 0; j < pool_size; j++) {
                            gi[(ph * pool_size + i) * W + (pw * pool_size + j)] = g;
                        }
                    }
                }
            }
        }
    }
}

/*
 * Transpose [C, B*S] -> [B, C*S]  (channel-major to batch-major)
 */
static void transpose_cb_to_bc(const float *src, float *dst,
                                int C, int B, int S) {
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            const float *s = src + c * B * S + b * S;
            float *d = dst + b * C * S + c * S;
            memcpy(d, s, S * sizeof(float));
        }
    }
}

/*
 * Transpose [B, C*S] -> [C, B*S]  (batch-major to channel-major)
 */
static void transpose_bc_to_cb(const float *src, float *dst,
                                int B, int C, int S) {
    for (int c = 0; c < C; c++) {
        for (int b = 0; b < B; b++) {
            const float *s = src + b * C * S + c * S;
            float *d = dst + c * B * S + b * S;
            memcpy(d, s, S * sizeof(float));
        }
    }
}

/*
 * Conv bias gradient: sum each row of [channels, spatial] -> [channels]
 */
static void conv_bias_grad(const float *grad, float *grad_bias,
                            int channels, int spatial) {
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        const float *gc = grad + c * spatial;
        for (int j = 0; j < spatial; j++)
            sum += gc[j];
        grad_bias[c] = sum;
    }
}

/* ------------------------------------------------------------------ */
/*  Initialize / Free                                                  */
/* ------------------------------------------------------------------ */

void cnn_initialize(CNN *cnn) {
    cnn->conv1_weights = (float *)malloc(C1_OUT * C1_COL_ROWS * sizeof(float));
    cnn->conv1_biases  = (float *)calloc(C1_OUT, sizeof(float));
    cnn->conv2_weights = (float *)malloc(C2_OUT * C2_COL_ROWS * sizeof(float));
    cnn->conv2_biases  = (float *)calloc(C2_OUT, sizeof(float));

    cnn->fc1_weights   = (float *)malloc(FLAT_SIZE * FC1_OUT * sizeof(float));
    cnn->fc1_biases    = (float *)calloc(FC1_OUT, sizeof(float));
    cnn->fc2_weights   = (float *)malloc(FC1_OUT * FC2_OUT * sizeof(float));
    cnn->fc2_biases    = (float *)calloc(FC2_OUT, sizeof(float));
    cnn->out_weights   = (float *)malloc(FC2_OUT * NUM_CLASSES * sizeof(float));
    cnn->out_biases    = (float *)calloc(NUM_CLASSES, sizeof(float));

    uint32_t rng = (uint32_t)time(NULL);
    nn_xavier_init(cnn->conv1_weights, C1_OUT * C1_COL_ROWS, C1_COL_ROWS, &rng);
    nn_xavier_init(cnn->conv2_weights, C2_OUT * C2_COL_ROWS, C2_COL_ROWS, &rng);
    nn_xavier_init(cnn->fc1_weights, FLAT_SIZE * FC1_OUT, FLAT_SIZE, &rng);
    nn_xavier_init(cnn->fc2_weights, FC1_OUT * FC2_OUT, FC1_OUT, &rng);
    nn_xavier_init(cnn->out_weights, FC2_OUT * NUM_CLASSES, FC2_OUT, &rng);
}

void cnn_free(CNN *cnn) {
    free(cnn->conv1_weights); free(cnn->conv1_biases);
    free(cnn->conv2_weights); free(cnn->conv2_biases);
    free(cnn->fc1_weights);   free(cnn->fc1_biases);
    free(cnn->fc2_weights);   free(cnn->fc2_biases);
    free(cnn->out_weights);   free(cnn->out_biases);
}

/* ------------------------------------------------------------------ */
/*  Batched forward pass: one GEMM per conv layer instead of B         */
/* ------------------------------------------------------------------ */

static void forward_batch(const CNN *cnn, const float *inputs, int bs,
                          float *col1_b, float *conv1_b, float *pool1_b,
                          float *col2_b, float *conv2_b, float *pool2_b,
                          float *flat,
                          float *fc1_out, float *fc2_out, float *out) {
    /*
     * Conv1: inputs[1, B*784] -> col1_b[25, B*576] -> conv1_b[6, B*576]
     *        -> bias + ReLU -> pool1_b[6, B*144]
     *
     * Since IN_C=1, inputs[B, 784] = [1, B*784] in channel-major.
     */
    im2col_batch(inputs, bs, C1_IN, IN_H, IN_W, C1_K, C1_K, 1, col1_b);
    /* conv1_b[6, B*576] = W1[6, 25] @ col1_b[25, B*576] */
    nn_sgemm_nn(C1_OUT, bs * C1_COL_COLS, C1_COL_ROWS,
                cnn->conv1_weights, col1_b, conv1_b);
    add_bias(conv1_b, cnn->conv1_biases, C1_OUT, bs * C1_COL_COLS);
    nn_relu_forward(conv1_b, C1_OUT * bs * C1_COL_COLS);
    avg_pool_forward_batch(conv1_b, pool1_b, C1_OUT, bs, C1_OH, C1_OW, P1_SIZE);

    /*
     * Conv2: pool1_b[6, B*144] -> col2_b[150, B*64] -> conv2_b[16, B*64]
     *        -> bias + ReLU -> pool2_b[16, B*16]
     */
    im2col_batch(pool1_b, bs, C2_IN, P1_H, P1_W, C2_K, C2_K, 1, col2_b);
    /* conv2_b[16, B*64] = W2[16, 150] @ col2_b[150, B*64] */
    nn_sgemm_nn(C2_OUT, bs * C2_COL_COLS, C2_COL_ROWS,
                cnn->conv2_weights, col2_b, conv2_b);
    add_bias(conv2_b, cnn->conv2_biases, C2_OUT, bs * C2_COL_COLS);
    nn_relu_forward(conv2_b, C2_OUT * bs * C2_COL_COLS);
    avg_pool_forward_batch(conv2_b, pool2_b, C2_OUT, bs, C2_OH, C2_OW, P2_SIZE);

    /* Transition: [16, B*16] -> flat[B, 256] */
    transpose_cb_to_bc(pool2_b, flat, C2_OUT, bs, P2_H * P2_W);

    /* FC1: flat[B, 256] @ fc1_weights[256, 120] -> fc1_out[B, 120] */
    nn_sgemm_nn(bs, FC1_OUT, FLAT_SIZE, flat, cnn->fc1_weights, fc1_out);
    nn_bias_relu(fc1_out, cnn->fc1_biases, bs, FC1_OUT);

    /* FC2: fc1_out[B, 120] @ fc2_weights[120, 84] -> fc2_out[B, 84] */
    nn_sgemm_nn(bs, FC2_OUT, FC1_OUT, fc1_out, cnn->fc2_weights, fc2_out);
    nn_bias_relu(fc2_out, cnn->fc2_biases, bs, FC2_OUT);

    /* Output: fc2_out[B, 84] @ out_weights[84, 10] -> out[B, 10] */
    nn_sgemm_nn(bs, NUM_CLASSES, FC2_OUT, fc2_out, cnn->out_weights, out);
    nn_bias_softmax(out, cnn->out_biases, bs, NUM_CLASSES);
}

/* ------------------------------------------------------------------ */
/*  Training: forward + backward + SGD                                */
/* ------------------------------------------------------------------ */

void cnn_train(CNN *cnn, float *inputs, int *targets, int num_samples,
               int batch_size, int num_epochs, float learning_rate) {

    int input_size = IN_C * IN_H * IN_W;

    /* Forward conv workspace (channel-major batched buffers) */
    float *col1_b  = (float *)malloc((size_t)C1_COL_ROWS * batch_size * C1_COL_COLS * sizeof(float));
    float *conv1_b = (float *)malloc((size_t)C1_OUT * batch_size * C1_COL_COLS * sizeof(float));
    float *pool1_b = (float *)malloc((size_t)C1_OUT * batch_size * P1_H * P1_W * sizeof(float));
    float *col2_b  = (float *)malloc((size_t)C2_COL_ROWS * batch_size * C2_COL_COLS * sizeof(float));
    float *conv2_b = (float *)malloc((size_t)C2_OUT * batch_size * C2_COL_COLS * sizeof(float));
    float *pool2_b = (float *)malloc((size_t)C2_OUT * batch_size * P2_H * P2_W * sizeof(float));

    /* FC layer buffers */
    float *flat    = (float *)malloc(batch_size * FLAT_SIZE * sizeof(float));
    float *fc1_out = (float *)malloc(batch_size * FC1_OUT * sizeof(float));
    float *fc2_out = (float *)malloc(batch_size * FC2_OUT * sizeof(float));
    float *out     = (float *)malloc(batch_size * NUM_CLASSES * sizeof(float));

    /* FC gradients */
    float *d_out    = (float *)malloc(batch_size * NUM_CLASSES * sizeof(float));
    float *d_fc2    = (float *)malloc(batch_size * FC2_OUT * sizeof(float));
    float *d_fc1    = (float *)malloc(batch_size * FC1_OUT * sizeof(float));
    float *d_flat   = (float *)malloc(batch_size * FLAT_SIZE * sizeof(float));

    float *grad_out_w = (float *)malloc(FC2_OUT * NUM_CLASSES * sizeof(float));
    float *grad_out_b = (float *)malloc(NUM_CLASSES * sizeof(float));
    float *grad_fc2_w = (float *)malloc(FC1_OUT * FC2_OUT * sizeof(float));
    float *grad_fc2_b = (float *)malloc(FC2_OUT * sizeof(float));
    float *grad_fc1_w = (float *)malloc(FLAT_SIZE * FC1_OUT * sizeof(float));
    float *grad_fc1_b = (float *)malloc(FC1_OUT * sizeof(float));

    /* Conv gradients (computed in one shot via batched GEMM) */
    float *grad_conv2_w = (float *)malloc(C2_OUT * C2_COL_ROWS * sizeof(float));
    float *grad_conv2_b = (float *)malloc(C2_OUT * sizeof(float));
    float *grad_conv1_w = (float *)malloc(C1_OUT * C1_COL_ROWS * sizeof(float));
    float *grad_conv1_b = (float *)malloc(C1_OUT * sizeof(float));

    /* Backward conv workspace (channel-major batched) */
    float *d_pool2_b = (float *)malloc((size_t)C2_OUT * batch_size * P2_H * P2_W * sizeof(float));
    float *d_conv2_b = (float *)malloc((size_t)C2_OUT * batch_size * C2_COL_COLS * sizeof(float));
    float *d_col2_b  = (float *)malloc((size_t)C2_COL_ROWS * batch_size * C2_COL_COLS * sizeof(float));
    float *d_pool1_b = (float *)malloc((size_t)C2_IN * batch_size * P1_H * P1_W * sizeof(float));
    float *d_conv1_b = (float *)malloc((size_t)C1_OUT * batch_size * C1_COL_COLS * sizeof(float));

    float *losses = (float *)malloc(batch_size * sizeof(float));

    int num_batches = (num_samples + batch_size - 1) / batch_size;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;

        for (int batch = 0; batch < num_batches; batch++) {
            int start = batch * batch_size;
            int bs = batch_size;
            if (start + bs > num_samples) bs = num_samples - start;

            const float *X = inputs + start * input_size;
            const int *Y = targets + start;

            /* ---- Forward (batched conv + batched FC) ---- */
            forward_batch(cnn, X, bs,
                          col1_b, conv1_b, pool1_b,
                          col2_b, conv2_b, pool2_b,
                          flat, fc1_out, fc2_out, out);

            /* ---- Loss + output gradient ---- */
            nn_cross_entropy_grad(out, Y, d_out, losses, bs, NUM_CLASSES);
            for (int i = 0; i < bs; i++)
                epoch_loss += losses[i];

            /* ---- FC backward (unchanged, already batched) ---- */

            /* grad_out_w = fc2_out^T @ d_out */
            nn_sgemm_tn(FC2_OUT, NUM_CLASSES, bs, fc2_out, d_out, grad_out_w);
            nn_col_sum(d_out, grad_out_b, bs, NUM_CLASSES);

            /* d_fc2 = d_out @ out_weights^T, then ReLU backward */
            nn_sgemm_nt(bs, FC2_OUT, NUM_CLASSES, d_out, cnn->out_weights, d_fc2);
            nn_relu_backward(d_fc2, fc2_out, bs * FC2_OUT);

            /* grad_fc2_w = fc1_out^T @ d_fc2 */
            nn_sgemm_tn(FC1_OUT, FC2_OUT, bs, fc1_out, d_fc2, grad_fc2_w);
            nn_col_sum(d_fc2, grad_fc2_b, bs, FC2_OUT);

            /* d_fc1 = d_fc2 @ fc2_weights^T, then ReLU backward */
            nn_sgemm_nt(bs, FC1_OUT, FC2_OUT, d_fc2, cnn->fc2_weights, d_fc1);
            nn_relu_backward(d_fc1, fc1_out, bs * FC1_OUT);

            /* grad_fc1_w = flat^T @ d_fc1 */
            nn_sgemm_tn(FLAT_SIZE, FC1_OUT, bs, flat, d_fc1, grad_fc1_w);
            nn_col_sum(d_fc1, grad_fc1_b, bs, FC1_OUT);

            /* d_flat = d_fc1 @ fc1_weights^T */
            nn_sgemm_nt(bs, FLAT_SIZE, FC1_OUT, d_fc1, cnn->fc1_weights, d_flat);

            /* ---- Conv backward (batched, one GEMM per layer) ---- */

            /* Transition: d_flat[B, 256] -> d_pool2_b[16, B*16] */
            transpose_bc_to_cb(d_flat, d_pool2_b, bs, C2_OUT, P2_H * P2_W);

            /* Pool2 backward: d_pool2_b[16, B*16] -> d_conv2_b[16, B*64] */
            avg_pool_backward_batch(d_pool2_b, d_conv2_b,
                                     C2_OUT, bs, C2_OH, C2_OW, P2_SIZE);

            /* ReLU backward for conv2 */
            nn_relu_backward(d_conv2_b, conv2_b, C2_OUT * bs * C2_COL_COLS);

            /* Conv2 weight gradient: d_conv2_b[16, B*64] @ col2_b^T[B*64, 150]
             * Automatically sums per-sample gradients via concatenated multiply. */
            nn_sgemm_nt(C2_OUT, C2_COL_ROWS, bs * C2_COL_COLS,
                        d_conv2_b, col2_b, grad_conv2_w);

            /* Conv2 bias gradient: sum each row of d_conv2_b */
            conv_bias_grad(d_conv2_b, grad_conv2_b, C2_OUT, bs * C2_COL_COLS);

            /* Conv2 input gradient: W2^T @ d_conv2_b -> d_col2_b */
            nn_sgemm_tn(C2_COL_ROWS, bs * C2_COL_COLS, C2_OUT,
                        cnn->conv2_weights, d_conv2_b, d_col2_b);

            /* col2im: d_col2_b[150, B*64] -> d_pool1_b[6, B*144] */
            memset(d_pool1_b, 0, (size_t)C2_IN * bs * P1_H * P1_W * sizeof(float));
            col2im_batch(d_col2_b, bs, C2_IN, P1_H, P1_W, C2_K, C2_K, 1, d_pool1_b);

            /* Pool1 backward: d_pool1_b[6, B*144] -> d_conv1_b[6, B*576] */
            avg_pool_backward_batch(d_pool1_b, d_conv1_b,
                                     C1_OUT, bs, C1_OH, C1_OW, P1_SIZE);

            /* ReLU backward for conv1 */
            nn_relu_backward(d_conv1_b, conv1_b, C1_OUT * bs * C1_COL_COLS);

            /* Conv1 weight gradient: d_conv1_b[6, B*576] @ col1_b^T[B*576, 25] */
            nn_sgemm_nt(C1_OUT, C1_COL_ROWS, bs * C1_COL_COLS,
                        d_conv1_b, col1_b, grad_conv1_w);

            /* Conv1 bias gradient */
            conv_bias_grad(d_conv1_b, grad_conv1_b, C1_OUT, bs * C1_COL_COLS);

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
            printf("  Epoch %4d  loss: %.4f\n", epoch, epoch_loss / num_samples);
            fflush(stdout);
        }
    }

    /* Free all workspace */
    free(col1_b); free(conv1_b); free(pool1_b);
    free(col2_b); free(conv2_b); free(pool2_b);
    free(flat); free(fc1_out); free(fc2_out); free(out);
    free(d_out); free(d_fc2); free(d_fc1); free(d_flat);
    free(grad_out_w); free(grad_out_b);
    free(grad_fc2_w); free(grad_fc2_b);
    free(grad_fc1_w); free(grad_fc1_b);
    free(grad_conv2_w); free(grad_conv2_b);
    free(grad_conv1_w); free(grad_conv1_b);
    free(d_pool2_b); free(d_conv2_b); free(d_col2_b);
    free(d_pool1_b); free(d_conv1_b);
    free(losses);
}

/* ------------------------------------------------------------------ */
/*  Evaluation                                                         */
/* ------------------------------------------------------------------ */

void cnn_evaluate(CNN *cnn, float *inputs, int *targets, int num_samples,
                  float *loss, float *accuracy) {
    int eval_bs = 256;

    float *col1_b  = (float *)malloc((size_t)C1_COL_ROWS * eval_bs * C1_COL_COLS * sizeof(float));
    float *conv1_b = (float *)malloc((size_t)C1_OUT * eval_bs * C1_COL_COLS * sizeof(float));
    float *pool1_b = (float *)malloc((size_t)C1_OUT * eval_bs * P1_H * P1_W * sizeof(float));
    float *col2_b  = (float *)malloc((size_t)C2_COL_ROWS * eval_bs * C2_COL_COLS * sizeof(float));
    float *conv2_b = (float *)malloc((size_t)C2_OUT * eval_bs * C2_COL_COLS * sizeof(float));
    float *pool2_b = (float *)malloc((size_t)C2_OUT * eval_bs * P2_H * P2_W * sizeof(float));
    float *flat    = (float *)malloc(eval_bs * FLAT_SIZE * sizeof(float));
    float *fc1_out = (float *)malloc(eval_bs * FC1_OUT * sizeof(float));
    float *fc2_out = (float *)malloc(eval_bs * FC2_OUT * sizeof(float));
    float *out     = (float *)malloc(eval_bs * NUM_CLASSES * sizeof(float));

    int input_size = IN_C * IN_H * IN_W;
    int num_batches = (num_samples + eval_bs - 1) / eval_bs;

    float total_loss = 0.0f;
    int correct = 0;

    for (int batch = 0; batch < num_batches; batch++) {
        int start = batch * eval_bs;
        int bs = eval_bs;
        if (start + bs > num_samples) bs = num_samples - start;

        forward_batch(cnn, inputs + start * input_size, bs,
                      col1_b, conv1_b, pool1_b,
                      col2_b, conv2_b, pool2_b,
                      flat, fc1_out, fc2_out, out);

        for (int i = 0; i < bs; i++) {
            const float *o = out + i * NUM_CLASSES;
            int label = targets[start + i];
            total_loss -= logf(o[label] + 1e-7f);

            int pred = 0;
            float mx = o[0];
            for (int j = 1; j < NUM_CLASSES; j++) {
                if (o[j] > mx) { mx = o[j]; pred = j; }
            }
            if (pred == label) correct++;
        }
    }

    *loss = total_loss / num_samples;
    *accuracy = (float)correct / num_samples * 100.0f;

    free(col1_b); free(conv1_b); free(pool1_b);
    free(col2_b); free(conv2_b); free(pool2_b);
    free(flat); free(fc1_out); free(fc2_out); free(out);
}
