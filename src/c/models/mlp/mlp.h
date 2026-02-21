#ifndef MLP_H
#define MLP_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Single hidden-layer MLP for multi-class classification.
 * Weights are stored in row-major order:
 *   input_weights[j * hidden_size + i]  = weight from input j to hidden i
 *   output_weights[j * output_size + i] = weight from hidden j to output i
 * Biases are initialized to zero; weights use Xavier uniform init.
 */
typedef struct {
    int input_size;
    int hidden_size;
    int output_size;

    float *input_weights;    /* [input_size  x hidden_size] */
    float *output_weights;   /* [hidden_size x output_size] */
    float *hidden_biases;    /* [hidden_size]               */
    float *output_biases;    /* [output_size]               */
} MLP;

/* Allocate parameters and apply Xavier uniform initialization */
void mlp_initialize(MLP *mlp, int input_size, int hidden_size, int output_size);

/* Free all parameter memory (heap or CUDA managed, depending on build) */
void mlp_free(MLP *mlp);

/* Mini-batch SGD training with cross-entropy loss */
void mlp_train(MLP *mlp, float *inputs, int *targets, int num_samples,
               int batch_size, int num_epochs, float learning_rate);

/* Compute average cross-entropy loss and accuracy (%) on a dataset */
void mlp_evaluate(MLP *mlp, float *inputs, int *targets, int num_samples, float *loss, float *accuracy);

#ifdef __cplusplus
}
#endif

#endif // MLP_H
