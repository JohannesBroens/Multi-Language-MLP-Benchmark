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
/*  im2col / col2im / pooling                                         */
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
/*  Per-sample forward pass through conv layers                       */
/*  Returns pointer to the flat vector (FLAT_SIZE) for FC layers      */
/* ------------------------------------------------------------------ */

/*
 * Workspace needed per sample for conv forward:
 *   col1:      C1_COL_ROWS * C1_COL_COLS  =  25 * 576  = 14400
 *   conv1_out: C1_OUT * C1_COL_COLS        =  6 * 576   = 3456
 *   pool1:     C1_OUT * P1_H * P1_W        =  6 * 144   = 864
 *   col2:      C2_COL_ROWS * C2_COL_COLS  = 150 * 64    = 9600
 *   conv2_out: C2_OUT * C2_COL_COLS        = 16 * 64    = 1024
 *   pool2:     FLAT_SIZE                   = 256
 */

#define WS_COL1     0
#define WS_CONV1    (C1_COL_ROWS * C1_COL_COLS)
#define WS_POOL1    (WS_CONV1 + C1_OUT * C1_COL_COLS)
#define WS_COL2     (WS_POOL1 + C1_OUT * P1_H * P1_W)
#define WS_CONV2    (WS_COL2 + C2_COL_ROWS * C2_COL_COLS)
#define WS_POOL2    (WS_CONV2 + C2_OUT * C2_COL_COLS)
#define WS_SIZE     (WS_POOL2 + FLAT_SIZE)

static void conv_forward_sample(const CNN *cnn, const float *input, float *ws) {
    float *col1     = ws + WS_COL1;
    float *conv1_out = ws + WS_CONV1;
    float *pool1    = ws + WS_POOL1;
    float *col2     = ws + WS_COL2;
    float *conv2_out = ws + WS_CONV2;
    float *pool2    = ws + WS_POOL2;

    /* Conv1: im2col -> GEMM -> bias + ReLU -> pool */
    im2col(input, C1_IN, IN_H, IN_W, C1_K, C1_K, 1, col1);
    /* conv1_out[6, 576] = conv1_weights[6, 25] @ col1[25, 576] */
    nn_sgemm_nn(C1_OUT, C1_COL_COLS, C1_COL_ROWS,
                cnn->conv1_weights, col1, conv1_out);
    add_bias(conv1_out, cnn->conv1_biases, C1_OUT, C1_COL_COLS);
    nn_relu_forward(conv1_out, C1_OUT * C1_COL_COLS);
    avg_pool_forward(conv1_out, pool1, C1_OUT, C1_OH, C1_OW, P1_SIZE);

    /* Conv2: im2col -> GEMM -> bias + ReLU -> pool */
    im2col(pool1, C2_IN, P1_H, P1_W, C2_K, C2_K, 1, col2);
    /* conv2_out[16, 64] = conv2_weights[16, 150] @ col2[150, 64] */
    nn_sgemm_nn(C2_OUT, C2_COL_COLS, C2_COL_ROWS,
                cnn->conv2_weights, col2, conv2_out);
    add_bias(conv2_out, cnn->conv2_biases, C2_OUT, C2_COL_COLS);
    nn_relu_forward(conv2_out, C2_OUT * C2_COL_COLS);
    avg_pool_forward(conv2_out, pool2, C2_OUT, C2_OH, C2_OW, P2_SIZE);

    /* pool2 is now the flat [FLAT_SIZE] vector for FC layers */
}

/* ------------------------------------------------------------------ */
/*  Batched forward pass (conv per-sample, FC batched via GEMM)       */
/* ------------------------------------------------------------------ */

static void forward_batch(const CNN *cnn, const float *inputs, int bs,
                          float *ws_all, float *flat,
                          float *fc1_out, float *fc2_out, float *out) {
    int input_size = IN_C * IN_H * IN_W;

    /* Per-sample conv layers */
    for (int b = 0; b < bs; b++) {
        const float *img = inputs + b * input_size;
        float *ws = ws_all + b * WS_SIZE;
        conv_forward_sample(cnn, img, ws);
        /* Copy pool2 output to flat batch matrix */
        memcpy(flat + b * FLAT_SIZE, ws + WS_POOL2, FLAT_SIZE * sizeof(float));
    }

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
/*  Training: forward + backward + SGD                                */
/* ------------------------------------------------------------------ */

void cnn_train(CNN *cnn, float *inputs, int *targets, int num_samples,
               int batch_size, int num_epochs, float learning_rate) {

    int input_size = IN_C * IN_H * IN_W;

    /* Allocate workspace for forward pass */
    float *ws_all  = (float *)malloc(batch_size * WS_SIZE * sizeof(float));
    float *flat    = (float *)malloc(batch_size * FLAT_SIZE * sizeof(float));
    float *fc1_out = (float *)malloc(batch_size * FC1_OUT * sizeof(float));
    float *fc2_out = (float *)malloc(batch_size * FC2_OUT * sizeof(float));
    float *out     = (float *)malloc(batch_size * NUM_CLASSES * sizeof(float));

    /* Gradients for FC layers */
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

    /* Gradients for conv layers (accumulated across batch) */
    float *grad_conv2_w = (float *)calloc(C2_OUT * C2_COL_ROWS, sizeof(float));
    float *grad_conv2_b = (float *)calloc(C2_OUT, sizeof(float));
    float *grad_conv1_w = (float *)calloc(C1_OUT * C1_COL_ROWS, sizeof(float));
    float *grad_conv1_b = (float *)calloc(C1_OUT, sizeof(float));

    /* Per-sample backward workspace for conv */
    float *d_pool2     = (float *)malloc(FLAT_SIZE * sizeof(float));
    float *d_conv2_out = (float *)malloc(C2_OUT * C2_COL_COLS * sizeof(float));
    float *d_col2      = (float *)malloc(C2_COL_ROWS * C2_COL_COLS * sizeof(float));
    float *d_pool1     = (float *)malloc(C2_IN * P1_H * P1_W * sizeof(float));
    float *d_conv1_out = (float *)malloc(C1_OUT * C1_COL_COLS * sizeof(float));

    /* Temp for per-sample weight gradients */
    float *tmp_grad_w2 = (float *)malloc(C2_OUT * C2_COL_ROWS * sizeof(float));
    float *tmp_grad_w1 = (float *)malloc(C1_OUT * C1_COL_ROWS * sizeof(float));

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

            /* ---- Forward ---- */
            forward_batch(cnn, X, bs, ws_all, flat, fc1_out, fc2_out, out);

            /* ---- Loss + output gradient ---- */
            nn_cross_entropy_grad(out, Y, d_out, losses, bs, NUM_CLASSES);
            for (int i = 0; i < bs; i++)
                epoch_loss += losses[i];

            /* ---- FC backward ---- */

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

            /* ---- Conv backward (per-sample) ---- */
            memset(grad_conv2_w, 0, C2_OUT * C2_COL_ROWS * sizeof(float));
            memset(grad_conv2_b, 0, C2_OUT * sizeof(float));
            memset(grad_conv1_w, 0, C1_OUT * C1_COL_ROWS * sizeof(float));
            memset(grad_conv1_b, 0, C1_OUT * sizeof(float));

            for (int b = 0; b < bs; b++) {
                float *ws = ws_all + b * WS_SIZE;
                float *col1      = ws + WS_COL1;
                float *conv1_out = ws + WS_CONV1;
                float *pool1     = ws + WS_POOL1;
                float *col2      = ws + WS_COL2;
                float *conv2_out = ws + WS_CONV2;

                /* d_flat[b] -> unflatten to d_pool2 [16, 4, 4] */
                memcpy(d_pool2, d_flat + b * FLAT_SIZE, FLAT_SIZE * sizeof(float));

                /* Pool2 backward: d_pool2[16,4,4] -> d_conv2_out[16,8,8] */
                memset(d_conv2_out, 0, C2_OUT * C2_COL_COLS * sizeof(float));
                avg_pool_backward(d_pool2, d_conv2_out, C2_OUT, C2_OH, C2_OW, P2_SIZE);

                /* ReLU backward for conv2 */
                nn_relu_backward(d_conv2_out, conv2_out, C2_OUT * C2_COL_COLS);

                /* Conv2 weight gradient: d_conv2_out[16,64] @ col2^T[64,150] */
                nn_sgemm_nt(C2_OUT, C2_COL_ROWS, C2_COL_COLS,
                            d_conv2_out, col2, tmp_grad_w2);
                for (int i = 0; i < C2_OUT * C2_COL_ROWS; i++)
                    grad_conv2_w[i] += tmp_grad_w2[i];

                /* Conv2 bias gradient: col_sum of d_conv2_out */
                for (int c = 0; c < C2_OUT; c++) {
                    float sum = 0.0f;
                    float *dc = d_conv2_out + c * C2_COL_COLS;
                    for (int j = 0; j < C2_COL_COLS; j++)
                        sum += dc[j];
                    grad_conv2_b[c] += sum;
                }

                /* Conv2 input gradient: conv2_weights^T @ d_conv2_out -> d_col2 */
                nn_sgemm_tn(C2_COL_ROWS, C2_COL_COLS, C2_OUT,
                            cnn->conv2_weights, d_conv2_out, d_col2);

                /* col2im: d_col2 -> d_pool1 [6, 12, 12] */
                memset(d_pool1, 0, C2_IN * P1_H * P1_W * sizeof(float));
                col2im(d_col2, C2_IN, P1_H, P1_W, C2_K, C2_K, 1, d_pool1);

                /* Pool1 backward: d_pool1[6,12,12] -> d_conv1_out[6,24,24] */
                memset(d_conv1_out, 0, C1_OUT * C1_COL_COLS * sizeof(float));
                avg_pool_backward(d_pool1, d_conv1_out, C1_OUT, C1_OH, C1_OW, P1_SIZE);

                /* ReLU backward for conv1 */
                nn_relu_backward(d_conv1_out, conv1_out, C1_OUT * C1_COL_COLS);

                /* Conv1 weight gradient: d_conv1_out[6,576] @ col1^T[576,25] */
                nn_sgemm_nt(C1_OUT, C1_COL_ROWS, C1_COL_COLS,
                            d_conv1_out, col1, tmp_grad_w1);
                for (int i = 0; i < C1_OUT * C1_COL_ROWS; i++)
                    grad_conv1_w[i] += tmp_grad_w1[i];

                /* Conv1 bias gradient */
                for (int c = 0; c < C1_OUT; c++) {
                    float sum = 0.0f;
                    float *dc = d_conv1_out + c * C1_COL_COLS;
                    for (int j = 0; j < C1_COL_COLS; j++)
                        sum += dc[j];
                    grad_conv1_b[c] += sum;
                }
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
            printf("  Epoch %4d  loss: %.4f\n", epoch, epoch_loss / num_samples);
            fflush(stdout);
        }
    }

    /* Free all workspace */
    free(ws_all); free(flat);
    free(fc1_out); free(fc2_out); free(out);
    free(d_out); free(d_fc2); free(d_fc1); free(d_flat);
    free(grad_out_w); free(grad_out_b);
    free(grad_fc2_w); free(grad_fc2_b);
    free(grad_fc1_w); free(grad_fc1_b);
    free(grad_conv2_w); free(grad_conv2_b);
    free(grad_conv1_w); free(grad_conv1_b);
    free(d_pool2); free(d_conv2_out); free(d_col2);
    free(d_pool1); free(d_conv1_out);
    free(tmp_grad_w2); free(tmp_grad_w1);
    free(losses);
}

/* ------------------------------------------------------------------ */
/*  Evaluation                                                         */
/* ------------------------------------------------------------------ */

void cnn_evaluate(CNN *cnn, float *inputs, int *targets, int num_samples,
                  float *loss, float *accuracy) {
    /* Process in batches to limit memory usage */
    int eval_bs = 256;
    float *ws_all  = (float *)malloc(eval_bs * WS_SIZE * sizeof(float));
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
                      ws_all, flat, fc1_out, fc2_out, out);

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

    free(ws_all); free(flat);
    free(fc1_out); free(fc2_out); free(out);
}
