#ifndef CNN_H
#define CNN_H

#include "nn_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * LeNet-5 CNN for MNIST digit classification.
 *
 * Architecture: Conv1(5x5,6) -> ReLU -> Pool(2x2) -> Conv2(5x5,16) -> ReLU -> Pool(2x2)
 *             -> Flatten(256) -> FC1(120) -> ReLU -> FC2(84) -> ReLU -> Output(10) -> Softmax
 *
 * Convolution is implemented via im2col: input patches are unfolded into columns of a
 * matrix, then the convolution becomes a standard GEMM call. This reuses the same
 * cache-tiled, OpenMP-parallelized GEMM from nn_ops, avoiding a separate convolution
 * implementation. The memory overhead of im2col (one extra matrix per conv layer) is
 * modest for LeNet-5's small filter sizes.
 *
 * Data layout: NCHW (batch, channels, height, width) throughout.
 * Flat indexing: data[b*C*H*W + c*H*W + h*W + w]
 */

typedef struct {
    /* Convolution layers -- weights stored as [C_out, C_in * kH * kW] for GEMM */
    float *conv1_weights;   /* [6, 1*5*5] = [6, 25] */
    float *conv1_biases;    /* [6] */
    float *conv2_weights;   /* [16, 6*5*5] = [16, 150] */
    float *conv2_biases;    /* [16] */

    /* Fully connected layers */
    float *fc1_weights;     /* [256, 120] */
    float *fc1_biases;      /* [120] */
    float *fc2_weights;     /* [120, 84] */
    float *fc2_biases;      /* [84] */
    float *out_weights;     /* [84, 10] */
    float *out_biases;      /* [10] */
} CNN;

/* Allocate and Xavier-initialize all parameters */
void cnn_initialize(CNN *cnn);

/* Free all parameter memory */
void cnn_free(CNN *cnn);

/* Mini-batch training with cross-entropy loss */
void cnn_train(CNN *cnn, float *inputs, int *targets, int num_samples,
               int batch_size, int num_epochs, float learning_rate,
               OptimizerType optimizer, SchedulerType scheduler);

/* Compute average loss and accuracy on a dataset */
void cnn_evaluate(CNN *cnn, float *inputs, int *targets, int num_samples,
                  float *loss, float *accuracy);

/* --- im2col / col2im utilities --- */

/*
 * im2col: unfold image patches into columns for GEMM-based convolution.
 *
 * For an input of shape (C_in, H, W) with filter (kH, kW), stride 1, no padding:
 *   output height OH = H - kH + 1
 *   output width  OW = W - kW + 1
 *   col matrix: [C_in * kH * kW,  OH * OW]
 *
 * Each column of the output is one flattened patch. The filter weight matrix
 * is [C_out, C_in * kH * kW]. Convolution = filters @ col_matrix.
 *
 * This is the same approach used by Caffe, PyTorch, and cuDNN internally.
 */
void im2col(const float *input, int C, int H, int W,
            int kH, int kW, int stride, float *col);

/* col2im: inverse of im2col -- scatter gradients back to spatial positions.
 * Accumulates (not overwrites) into output, handling overlapping patches. */
void col2im(const float *col, int C, int H, int W,
            int kH, int kW, int stride, float *output);

/* Average pooling forward: pools each pool_size x pool_size window.
 * Input: [C, H, W], Output: [C, H/pool_size, W/pool_size] */
void avg_pool_forward(const float *input, float *output,
                      int C, int H, int W, int pool_size);

/* Average pooling backward: distribute gradient equally to pooled positions */
void avg_pool_backward(const float *grad_output, float *grad_input,
                       int C, int H, int W, int pool_size);

#ifdef __cplusplus
}
#endif

#endif /* CNN_H */
