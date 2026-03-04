#ifndef MLP_H
#define MLP_H

#include "nn_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Multi-hidden-layer MLP for multi-class classification.
 * Weights are stored in row-major order:
 *   weights[i][r * layer_sizes[i+1] + c] = weight from layer i neuron r to layer i+1 neuron c
 * Biases are initialized to zero; weights use Xavier uniform init.
 *
 * layer_sizes = [input_size, hidden_size, ..., hidden_size, output_size]
 * num_layers  = num_hidden_layers + 1  (total weight matrices)
 */
typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    int num_layers;      /* num_hidden_layers + 1 (total weight matrices) */
    int *layer_sizes;    /* [input, hid, ..., hid, output], length num_layers+1 */
    float **weights;     /* weights[i]: [layer_sizes[i] x layer_sizes[i+1]] */
    float **biases;      /* biases[i]:  [layer_sizes[i+1]] */
} MLP;

/* Allocate parameters and apply Xavier uniform initialization */
void mlp_initialize(MLP *mlp, int input_size, int hidden_size, int output_size,
                    int num_hidden_layers);

/* Free all parameter memory (heap or CUDA managed, depending on build) */
void mlp_free(MLP *mlp);

/* Mini-batch training with cross-entropy loss */
void mlp_train(MLP *mlp, float *inputs, int *targets, int num_samples,
               int batch_size, int num_epochs, float learning_rate,
               OptimizerType optimizer, SchedulerType scheduler);

/* Compute average cross-entropy loss and accuracy (%) on a dataset */
void mlp_evaluate(MLP *mlp, float *inputs, int *targets, int num_samples, float *loss, float *accuracy);

#ifdef __cplusplus
}
#endif

#endif // MLP_H
