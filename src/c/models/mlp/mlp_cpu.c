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

void mlp_initialize(MLP *mlp, int input_size, int hidden_size, int output_size,
                    int num_hidden_layers) {
    mlp->input_size = input_size;
    mlp->hidden_size = hidden_size;
    mlp->output_size = output_size;
    mlp->num_layers = num_hidden_layers + 1;

    int nl = mlp->num_layers;

    /* Build layer_sizes: [input, hid, ..., hid, output] */
    mlp->layer_sizes = (int *)malloc((nl + 1) * sizeof(int));
    mlp->layer_sizes[0] = input_size;
    for (int i = 1; i <= num_hidden_layers; ++i)
        mlp->layer_sizes[i] = hidden_size;
    mlp->layer_sizes[nl] = output_size;

    /* Allocate weight and bias arrays */
    mlp->weights = (float **)malloc(nl * sizeof(float *));
    mlp->biases  = (float **)malloc(nl * sizeof(float *));

    uint32_t rng = (uint32_t)time(NULL);

    for (int i = 0; i < nl; ++i) {
        int fan_in  = mlp->layer_sizes[i];
        int fan_out = mlp->layer_sizes[i + 1];
        mlp->weights[i] = (float *)malloc(fan_in * fan_out * sizeof(float));
        mlp->biases[i]  = (float *)calloc(fan_out, sizeof(float));

        if (!mlp->weights[i] || !mlp->biases[i]) {
            fprintf(stderr, "Failed to allocate MLP parameters for layer %d.\n", i);
            exit(EXIT_FAILURE);
        }

        nn_xavier_init(mlp->weights[i], fan_in * fan_out, fan_in, &rng);
    }
}

void mlp_free(MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; ++i) {
        free(mlp->weights[i]);
        free(mlp->biases[i]);
    }
    free(mlp->weights);
    free(mlp->biases);
    free(mlp->layer_sizes);
}

/* ------------------------------------------------------------------ */
/*  Batched forward pass                                              */
/* ------------------------------------------------------------------ */

/*
 * Forward pass storing all intermediate activations for backprop.
 * acts[0] = hidden layer 0, ..., acts[num_hidden-1] = last hidden layer
 * output = final softmax output
 */
static void forward_batch(const MLP *mlp, const float *X, int bs,
                          float **acts, float *output) {
    int nl = mlp->num_layers;
    int num_hidden = nl - 1;

    for (int i = 0; i < nl; ++i) {
        int in_dim  = mlp->layer_sizes[i];
        int out_dim = mlp->layer_sizes[i + 1];
        const float *input_act = (i == 0) ? X : acts[i - 1];
        float *out_act = (i < num_hidden) ? acts[i] : output;

        nn_sgemm_nn(bs, out_dim, in_dim, input_act, mlp->weights[i], out_act);

        if (i < num_hidden) {
            nn_bias_relu(out_act, mlp->biases[i], bs, out_dim);
        } else {
            nn_bias_softmax(out_act, mlp->biases[i], bs, out_dim);
        }
    }
}

/* Forward pass for evaluation (no activation storage needed for single hidden) */
static void forward_eval(const MLP *mlp, const float *X, int bs, float *output) {
    int nl = mlp->num_layers;
    int num_hidden = nl - 1;

    /* Allocate temp buffers for hidden activations */
    float *cur = NULL, *prev = NULL;
    int max_hid = mlp->hidden_size;

    if (num_hidden > 0) {
        cur  = (float *)malloc(bs * max_hid * sizeof(float));
        prev = (float *)malloc(bs * max_hid * sizeof(float));
    }

    for (int i = 0; i < nl; ++i) {
        int in_dim  = mlp->layer_sizes[i];
        int out_dim = mlp->layer_sizes[i + 1];
        const float *input_act;
        float *out_act;

        if (i == 0) {
            input_act = X;
        } else if (i % 2 == 1) {
            input_act = cur;
        } else {
            input_act = prev;
        }

        if (i < num_hidden) {
            out_act = (i % 2 == 0) ? cur : prev;
        } else {
            out_act = output;
        }

        nn_sgemm_nn(bs, out_dim, in_dim, input_act, mlp->weights[i], out_act);

        if (i < num_hidden) {
            nn_bias_relu(out_act, mlp->biases[i], bs, out_dim);
        } else {
            nn_bias_softmax(out_act, mlp->biases[i], bs, out_dim);
        }
    }

    free(cur);
    free(prev);
}

/* ------------------------------------------------------------------ */
/*  Training: batched GEMM forward + backward                        */
/* ------------------------------------------------------------------ */

void mlp_train(MLP *mlp, float *inputs, int *targets, int num_samples,
               int batch_size, int num_epochs, float learning_rate,
               OptimizerType optimizer, SchedulerType scheduler) {
    int nl = mlp->num_layers;
    int num_hidden = nl - 1;
    int hid = mlp->hidden_size;
    int out = mlp->output_size;

    /* Allocate hidden activation buffers for forward pass */
    float **acts = (float **)malloc(num_hidden * sizeof(float *));
    for (int i = 0; i < num_hidden; ++i)
        acts[i] = (float *)malloc(batch_size * hid * sizeof(float));

    float *output   = (float *)malloc(batch_size * out * sizeof(float));
    float *d_output = (float *)malloc(batch_size * out * sizeof(float));
    float *losses   = (float *)malloc(batch_size * sizeof(float));

    /* d_current and d_prev for gradient propagation */
    int max_dim = hid > out ? hid : out;
    float *d_current = (float *)malloc(batch_size * max_dim * sizeof(float));
    float *d_prev    = (float *)malloc(batch_size * max_dim * sizeof(float));

    /* Per-layer gradient buffers */
    float **grad_W = (float **)malloc(nl * sizeof(float *));
    float **grad_b = (float **)malloc(nl * sizeof(float *));
    for (int i = 0; i < nl; ++i) {
        int fan_in  = mlp->layer_sizes[i];
        int fan_out = mlp->layer_sizes[i + 1];
        grad_W[i] = (float *)malloc(fan_in * fan_out * sizeof(float));
        grad_b[i] = (float *)malloc(fan_out * sizeof(float));
    }

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

            const float *X = inputs + start * mlp->input_size;
            const int *Y   = targets + start;

            /* ---- Forward ---- */
            forward_batch(mlp, X, bs, acts, output);

            /* ---- Loss + d_output = softmax - one_hot ---- */
            nn_cross_entropy_grad(output, Y, d_output, losses, bs, out);
            for (int i = 0; i < bs; ++i)
                epoch_loss += losses[i];

            /* ---- Backward ---- */
            /* Start with d_output as the current gradient */
            float *dcur = d_output;

            for (int layer = nl - 1; layer >= 0; --layer) {
                int fan_in  = mlp->layer_sizes[layer];
                int fan_out = mlp->layer_sizes[layer + 1];
                const float *input_act = (layer == 0) ? X : acts[layer - 1];

                /* grad_W[layer] = input_act^T @ dcur */
                nn_sgemm_tn(fan_in, fan_out, bs, input_act, dcur, grad_W[layer]);

                /* grad_b[layer] = col_sum(dcur) */
                nn_col_sum(dcur, grad_b[layer], bs, fan_out);

                /* Propagate gradient to previous layer (skip for layer 0) */
                if (layer > 0) {
                    /* d_prev = dcur @ W[layer]^T */
                    float *dnext = (dcur == d_output) ? d_current :
                                   (dcur == d_current) ? d_prev : d_current;
                    nn_sgemm_nt(bs, fan_in, fan_out, dcur, mlp->weights[layer], dnext);
                    /* ReLU backward */
                    nn_relu_backward(dnext, acts[layer - 1], bs * fan_in);
                    dcur = dnext;
                }
            }

            /* ---- Parameter update ---- */
            float lr_s = lr / (float)bs;
            step++;

            for (int i = 0; i < nl; ++i) {
                int fan_in  = mlp->layer_sizes[i];
                int fan_out = mlp->layer_sizes[i + 1];
                if (optimizer == OPT_ADAM) {
                    nn_adam_update(mlp->weights[i], grad_W[i], &adam_W[i], lr, beta1, beta2, eps, step);
                    nn_adam_update(mlp->biases[i], grad_b[i], &adam_b[i], lr, beta1, beta2, eps, step);
                } else {
                    nn_sgd_update(mlp->weights[i], grad_W[i], lr_s, fan_in * fan_out);
                    nn_sgd_update(mlp->biases[i], grad_b[i], lr_s, fan_out);
                }
            }
        }

        if (epoch % 100 == 0)
            printf("  Epoch %4d  loss: %.4f\n", epoch, epoch_loss / num_samples);
    }

    if (optimizer == OPT_ADAM) {
        for (int i = 0; i < nl; ++i) {
            nn_adam_state_free(&adam_W[i]);
            nn_adam_state_free(&adam_b[i]);
        }
        free(adam_W);
        free(adam_b);
    }

    for (int i = 0; i < num_hidden; ++i)
        free(acts[i]);
    free(acts);
    free(output);
    free(d_output);
    free(losses);
    free(d_current);
    free(d_prev);
    for (int i = 0; i < nl; ++i) {
        free(grad_W[i]);
        free(grad_b[i]);
    }
    free(grad_W);
    free(grad_b);
}

/* ------------------------------------------------------------------ */
/*  Evaluation: batched forward                                       */
/* ------------------------------------------------------------------ */

void mlp_evaluate(MLP *mlp, float *inputs, int *targets, int num_samples,
                  float *loss, float *accuracy) {
    int out = mlp->output_size;

    float *output = (float *)malloc(num_samples * out * sizeof(float));
    if (!output) {
        fprintf(stderr, "Failed to allocate evaluation workspace.\n");
        exit(EXIT_FAILURE);
    }

    forward_eval(mlp, inputs, num_samples, output);

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

    free(output);
}
