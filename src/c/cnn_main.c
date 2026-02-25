#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "cnn.h"
#include "data_loader.h"

static void print_usage(const char *prog) {
    printf("Usage: %s --dataset [mnist] [options]\n", prog);
    printf("  --batch-size N    Mini-batch size (default: 64)\n");
    printf("  --epochs N        Number of training epochs (default: 20)\n");
}

/* Fisher-Yates shuffle: randomly permute samples in-place. */
static void shuffle_data(float *inputs, int *labels, int num_samples, int input_size) {
    for (int i = num_samples - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int tmp_label = labels[i];
        labels[i] = labels[j];
        labels[j] = tmp_label;
        for (int k = 0; k < input_size; ++k) {
            float tmp = inputs[i * input_size + k];
            inputs[i * input_size + k] = inputs[j * input_size + k];
            inputs[j * input_size + k] = tmp;
        }
    }
}

/* CLOCK_MONOTONIC wall-clock timer */
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    char *dataset_name = "mnist";
    int batch_size = 64;
    int num_epochs = 20;
    float learning_rate = 0.01f;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--dataset") == 0 || strcmp(argv[i], "-d") == 0) && i + 1 < argc) {
            dataset_name = argv[++i];
        } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            num_epochs = atoi(argv[++i]);
        } else {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    float *inputs = NULL;
    int *labels = NULL;
    int num_samples = 0;
    int input_size = 0;
    int output_size = 0;

    /* Load dataset -- CNN expects MNIST (28x28 images) */
    int err = -1;
    if (strcmp(dataset_name, "mnist") == 0) {
        err = load_mnist_data(&inputs, &labels, &num_samples, &input_size, &output_size, 0);
    } else {
        fprintf(stderr, "CNN only supports MNIST dataset. Got: %s\n", dataset_name);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (err != 0) {
        fprintf(stderr, "Failed to load dataset: %s\n", dataset_name);
        return EXIT_FAILURE;
    }

    printf("Dataset: %s  (%d samples, %d features, %d classes)\n",
           dataset_name, num_samples, input_size, output_size);
    fflush(stdout);

    /* No z-score normalization for MNIST (pixels already [0,1]) */

    /* Shuffle then split 80/20 */
    srand((unsigned int)time(NULL));
    shuffle_data(inputs, labels, num_samples, input_size);

    int train_size = (int)(num_samples * 0.8f);
    int test_size = num_samples - train_size;
    float *train_inputs = inputs;
    int *train_labels = labels;
    float *test_inputs = inputs + train_size * input_size;
    int *test_labels = labels + train_size;

    printf("Train: %d samples, Test: %d samples\n", train_size, test_size);

    CNN cnn;
    cnn_initialize(&cnn);

    printf("\nTraining LeNet-5 (%d epochs, batch_size=%d, lr=%.4f)...\n",
           num_epochs, batch_size, learning_rate);

    double t_start = get_time_sec();
    cnn_train(&cnn, train_inputs, train_labels, train_size, batch_size, num_epochs, learning_rate);
    double t_train = get_time_sec() - t_start;

    float loss, accuracy;
    double t_eval_start = get_time_sec();
    cnn_evaluate(&cnn, test_inputs, test_labels, test_size, &loss, &accuracy);
    double t_eval = get_time_sec() - t_eval_start;

    /* Standardized output format -- parsed by benchmark.py */
    printf("\n=== Results on %s ===\n", dataset_name);
    printf("Test Loss:     %.4f\n", loss);
    printf("Test Accuracy: %.2f%%\n", accuracy);
    printf("Train time:    %.3f s\n", t_train);
    printf("Eval time:     %.3f s\n", t_eval);
    printf("Throughput:    %.0f samples/s\n", (double)train_size * num_epochs / t_train);

    cnn_free(&cnn);
    free(inputs);
    free(labels);

    return EXIT_SUCCESS;
}
