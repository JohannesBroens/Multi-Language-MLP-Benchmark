#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * All loaders allocate flat arrays via malloc and return dimensions via output params.
 * Caller is responsible for free(). Returns 0 on success, -1 on failure.
 *
 * inputs:  [num_samples x input_size]  row-major float array
 * labels:  [num_samples]               int array of class indices
 */

/* Synthetic 2D circle data (2 features, 2 classes). num_samples_requested <= 0 defaults to 1000. */
int load_generated_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size, int num_samples_requested);

/* UCI Iris (data/iris_processed.txt — 150 samples, 4 features, 3 classes) */
int load_iris_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size);

/* UCI Wine Quality (semicolon CSV — 11 features, 11 classes for scores 0-10) */
int load_wine_quality_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size, const char *filename);

/* Wisconsin Breast Cancer Diagnostic (data/wdbc.data — 30 features, 2 classes) */
int load_breast_cancer_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size);

/* MNIST handwritten digits (data/train-images-idx3-ubyte — 60K images, 28x28, 10 classes)
 * Reads IDX binary format. Pixel values normalized to [0, 1].
 * If is_test != 0, loads the 10K test set instead. */
int load_mnist_data(float **inputs, int **labels, int *num_samples,
                    int *input_size, int *output_size, int is_test);

#ifdef __cplusplus
}
#endif

#endif // DATA_LOADER_H
