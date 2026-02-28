#include "mlp.h"
#include "nn_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  Init / Free                                                       */
/* ------------------------------------------------------------------ */

void mlp_initialize(MLP *mlp, int input_size, int hidden_size, int output_size) {
    mlp->input_size = input_size;
    mlp->hidden_size = hidden_size;
    mlp->output_size = output_size;

    mlp->input_weights  = (float *)malloc(input_size  * hidden_size * sizeof(float));
    mlp->output_weights = (float *)malloc(hidden_size * output_size * sizeof(float));
    mlp->hidden_biases  = (float *)calloc(hidden_size, sizeof(float));
    mlp->output_biases  = (float *)calloc(output_size, sizeof(float));

    if (!mlp->input_weights || !mlp->output_weights ||
        !mlp->hidden_biases || !mlp->output_biases) {
        fprintf(stderr, "Failed to allocate MLP parameters.\n");
        exit(EXIT_FAILURE);
    }

    uint32_t rng = (uint32_t)time(NULL);
    nn_xavier_init(mlp->input_weights, input_size * hidden_size, input_size, &rng);
    nn_xavier_init(mlp->output_weights, hidden_size * output_size, hidden_size, &rng);
}

void mlp_free(MLP *mlp) {
    free(mlp->input_weights);
    free(mlp->output_weights);
    free(mlp->hidden_biases);
    free(mlp->output_biases);
}

/* ------------------------------------------------------------------ */
/*  Batched forward pass                                              */
/* ------------------------------------------------------------------ */

static void forward_batch(const MLP *mlp, const float *X, int bs,
                          float *hidden, float *output) {
    int in  = mlp->input_size;
    int hid = mlp->hidden_size;
    int out = mlp->output_size;

    /* hidden = X @ W1 */
    nn_sgemm_nn(bs, hid, in, X, mlp->input_weights, hidden);

    /* Add bias + ReLU */
    nn_bias_relu(hidden, mlp->hidden_biases, bs, hid);

    /* output = hidden @ W2 */
    nn_sgemm_nn(bs, out, hid, hidden, mlp->output_weights, output);

    /* Add bias + softmax (per sample) */
    nn_bias_softmax(output, mlp->output_biases, bs, out);
}

/* ------------------------------------------------------------------ */
/*  Training: batched GEMM forward + backward                        */
/* ------------------------------------------------------------------ */

void mlp_train(MLP *mlp, float *inputs, int *targets, int num_samples,
               int batch_size, int num_epochs, float learning_rate,
               OptimizerType optimizer, SchedulerType scheduler) {
    int in  = mlp->input_size;
    int hid = mlp->hidden_size;
    int out = mlp->output_size;

    float *hidden   = (float *)malloc(batch_size * hid * sizeof(float));
    float *output   = (float *)malloc(batch_size * out * sizeof(float));
    float *d_output = (float *)malloc(batch_size * out * sizeof(float));
    float *d_hidden = (float *)malloc(batch_size * hid * sizeof(float));
    float *grad_W1  = (float *)malloc(in  * hid * sizeof(float));
    float *grad_W2  = (float *)malloc(hid * out * sizeof(float));
    float *grad_b1  = (float *)malloc(hid * sizeof(float));
    float *grad_b2  = (float *)malloc(out * sizeof(float));
    float *losses   = (float *)malloc(batch_size * sizeof(float));

    if (!hidden || !output || !d_output || !d_hidden ||
        !grad_W1 || !grad_W2 || !grad_b1 || !grad_b2 || !losses) {
        fprintf(stderr, "Failed to allocate training workspace.\n");
        exit(EXIT_FAILURE);
    }

    /* Adam optimizer state */
    AdamState adam_W1 = {0}, adam_W2 = {0}, adam_b1 = {0}, adam_b2 = {0};
    int step = 0;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    if (optimizer == OPT_ADAM) {
        nn_adam_state_init(&adam_W1, in * hid);
        nn_adam_state_init(&adam_W2, hid * out);
        nn_adam_state_init(&adam_b1, hid);
        nn_adam_state_init(&adam_b2, out);
    }

    int num_batches = (num_samples + batch_size - 1) / batch_size;
    int warmup_epochs = (scheduler == SCHED_COSINE) ? (int)(num_epochs * 0.05f) : 0;
    if (warmup_epochs < 1 && scheduler == SCHED_COSINE) warmup_epochs = 1;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float lr = (scheduler == SCHED_COSINE)
            ? nn_cosine_lr(epoch, num_epochs, learning_rate, warmup_epochs, 1e-6f)
            : learning_rate;
        float epoch_loss = 0.0f;

        for (int batch = 0; batch < num_batches; ++batch) {
            int start = batch * batch_size;
            int bs = batch_size;
            if (start + bs > num_samples) bs = num_samples - start;

            const float *X = inputs + start * in;
            const int *Y   = targets + start;

            /* ---- Forward ---- */
            forward_batch(mlp, X, bs, hidden, output);

            /* ---- Loss + d_output = softmax - one_hot ---- */
            nn_cross_entropy_grad(output, Y, d_output, losses, bs, out);
            for (int i = 0; i < bs; ++i)
                epoch_loss += losses[i];

            /* ---- Backward ---- */

            /* grad_W2[hid,out] = hidden^T @ d_output */
            nn_sgemm_tn(hid, out, bs, hidden, d_output, grad_W2);

            /* grad_b2 = column sum of d_output */
            nn_col_sum(d_output, grad_b2, bs, out);

            /* d_hidden = d_output @ W2^T, then ReLU mask */
            nn_sgemm_nt(bs, hid, out, d_output, mlp->output_weights, d_hidden);
            nn_relu_backward(d_hidden, hidden, bs * hid);

            /* grad_W1[in,hid] = X^T @ d_hidden */
            nn_sgemm_tn(in, hid, bs, X, d_hidden, grad_W1);

            /* grad_b1 = column sum of d_hidden */
            nn_col_sum(d_hidden, grad_b1, bs, hid);

            /* ---- Parameter update ---- */
            float lr_s = lr / (float)bs;
            step++;

            if (optimizer == OPT_ADAM) {
                nn_adam_update(mlp->input_weights, grad_W1, &adam_W1, lr_s, beta1, beta2, eps, step);
                nn_adam_update(mlp->output_weights, grad_W2, &adam_W2, lr_s, beta1, beta2, eps, step);
                nn_adam_update(mlp->hidden_biases, grad_b1, &adam_b1, lr_s, beta1, beta2, eps, step);
                nn_adam_update(mlp->output_biases, grad_b2, &adam_b2, lr_s, beta1, beta2, eps, step);
            } else {
                nn_sgd_update(mlp->input_weights, grad_W1, lr_s, in * hid);
                nn_sgd_update(mlp->output_weights, grad_W2, lr_s, hid * out);
                nn_sgd_update(mlp->hidden_biases, grad_b1, lr_s, hid);
                nn_sgd_update(mlp->output_biases, grad_b2, lr_s, out);
            }
        }

        if (epoch % 100 == 0)
            printf("  Epoch %4d  loss: %.4f\n", epoch, epoch_loss / num_samples);
    }

    if (optimizer == OPT_ADAM) {
        nn_adam_state_free(&adam_W1);
        nn_adam_state_free(&adam_W2);
        nn_adam_state_free(&adam_b1);
        nn_adam_state_free(&adam_b2);
    }

    free(hidden);  free(output);
    free(d_output); free(d_hidden);
    free(grad_W1); free(grad_W2);
    free(grad_b1); free(grad_b2);
    free(losses);
}

/* ------------------------------------------------------------------ */
/*  Evaluation: batched forward                                       */
/* ------------------------------------------------------------------ */

void mlp_evaluate(MLP *mlp, float *inputs, int *targets, int num_samples,
                  float *loss, float *accuracy) {
    int hid = mlp->hidden_size;
    int out = mlp->output_size;

    float *hidden = (float *)malloc(num_samples * hid * sizeof(float));
    float *output = (float *)malloc(num_samples * out * sizeof(float));

    if (!hidden || !output) {
        fprintf(stderr, "Failed to allocate evaluation workspace.\n");
        exit(EXIT_FAILURE);
    }

    forward_batch(mlp, inputs, num_samples, hidden, output);

    float total_loss = 0.0f;
    int correct = 0;

    for (int i = 0; i < num_samples; ++i) {
        const float *o = output + i * out;
        total_loss -= logf(o[targets[i]] + 1e-7f);

        int pred = 0;
        float mx = o[0];
        for (int j = 1; j < out; ++j) {
            if (o[j] > mx) { mx = o[j]; pred = j; }
        }
        if (pred == targets[i]) correct++;
    }

    *loss = total_loss / num_samples;
    *accuracy = (float)correct / num_samples * 100.0f;

    free(hidden);
    free(output);
}
