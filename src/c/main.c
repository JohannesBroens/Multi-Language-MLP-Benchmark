#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mlp.h"
#include "data_loader.h"

static void print_usage(const char *prog) {
    printf("Usage: %s --dataset [generated|iris|wine-red|wine-white|breast-cancer|mnist]\n", prog);
    printf("  --batch-size N    Mini-batch size (default: 32)\n");
    printf("  --num-samples N   Number of samples for generated dataset (default: 1000)\n");
    printf("  --hidden-size N   Hidden layer size (default: 64)\n");
    printf("  --epochs N        Number of training epochs (default: 1000)\n");
}

/*
 * Z-score normalization per feature: x' = (x - mean) / std.
 * Epsilon 1e-8 prevents division by zero for constant features.
 */
static void normalize_features(float *inputs, int num_samples, int input_size) {
    for (int j = 0; j < input_size; ++j) {
        float mean = 0.0f;
        for (int i = 0; i < num_samples; ++i)
            mean += inputs[i * input_size + j];
        mean /= num_samples;

        float var = 0.0f;
        for (int i = 0; i < num_samples; ++i) {
            float d = inputs[i * input_size + j] - mean;
            var += d * d;
        }
        float std = sqrtf(var / num_samples + 1e-8f);

        for (int i = 0; i < num_samples; ++i)
            inputs[i * input_size + j] = (inputs[i * input_size + j] - mean) / std;
    }
}

/* Fisher-Yates shuffle: randomly permute samples in-place.
 * Swaps both input feature rows and corresponding labels. */
static void shuffle_data(float *inputs, int *labels, int num_samples, int input_size) {
    for (int i = num_samples - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        /* Swap labels */
        int tmp_label = labels[i];
        labels[i] = labels[j];
        labels[j] = tmp_label;
        /* Swap input rows */
        for (int k = 0; k < input_size; ++k) {
            float tmp = inputs[i * input_size + k];
            inputs[i * input_size + k] = inputs[j * input_size + k];
            inputs[j * input_size + k] = tmp;
        }
    }
}

/* CLOCK_MONOTONIC wall-clock timer for benchmarking (not affected by NTP adjustments) */
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    char *dataset_name = NULL;
    int batch_size = 32;
    int num_samples_requested = 0;  /* 0 = use dataset default */
    int hidden_size = 64;
    int num_epochs = 1000;
    float learning_rate = 0.01f;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--dataset") == 0 || strcmp(argv[i], "-d") == 0) && i + 1 < argc) {
            dataset_name = argv[++i];
        } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--num-samples") == 0 && i + 1 < argc) {
            num_samples_requested = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden-size") == 0 && i + 1 < argc) {
            hidden_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            num_epochs = atoi(argv[++i]);
        } else {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (dataset_name == NULL) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    float *inputs = NULL;
    int *labels = NULL;
    int num_samples = 0;
    int input_size = 0;
    int output_size = 0;

    /* Dispatch to the appropriate data loader */
    int err = -1;
    if (strcmp(dataset_name, "generated") == 0) {
        err = load_generated_data(&inputs, &labels, &num_samples, &input_size, &output_size, num_samples_requested);
    } else if (strcmp(dataset_name, "iris") == 0) {
        err = load_iris_data(&inputs, &labels, &num_samples, &input_size, &output_size);
    } else if (strcmp(dataset_name, "wine-red") == 0) {
        err = load_wine_quality_data(&inputs, &labels, &num_samples, &input_size, &output_size, "data/winequality-red.csv");
    } else if (strcmp(dataset_name, "wine-white") == 0) {
        err = load_wine_quality_data(&inputs, &labels, &num_samples, &input_size, &output_size, "data/winequality-white.csv");
    } else if (strcmp(dataset_name, "breast-cancer") == 0) {
        err = load_breast_cancer_data(&inputs, &labels, &num_samples, &input_size, &output_size);
    } else if (strcmp(dataset_name, "mnist") == 0) {
        err = load_mnist_data(&inputs, &labels, &num_samples, &input_size, &output_size, 0);
    } else {
        fprintf(stderr, "Unknown dataset: %s\n", dataset_name);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (err != 0) {
        fprintf(stderr, "Failed to load dataset: %s\n", dataset_name);
        return EXIT_FAILURE;
    }

    printf("Dataset: %s  (%d samples, %d features, %d classes)\n",
           dataset_name, num_samples, input_size, output_size);

    /* Skip z-score normalization for MNIST (pixels already [0,1]) */
    if (strcmp(dataset_name, "mnist") != 0)
        normalize_features(inputs, num_samples, input_size);

    /* Shuffle then split 80/20 — test set is a pointer offset (zero-copy) */
    srand((unsigned int)time(NULL));
    shuffle_data(inputs, labels, num_samples, input_size);

    int train_size = (int)(num_samples * 0.8f);
    int test_size = num_samples - train_size;
    float *train_inputs = inputs;
    int *train_labels = labels;
    float *test_inputs = inputs + train_size * input_size;
    int *test_labels = labels + train_size;

    printf("Train: %d samples, Test: %d samples\n", train_size, test_size);

    MLP mlp;
    mlp_initialize(&mlp, input_size, hidden_size, output_size);

    printf("\nTraining (%d epochs, batch_size=%d, hidden=%d, lr=%.4f)...\n",
           num_epochs, batch_size, hidden_size, learning_rate);

    double t_start = get_time_sec();
    mlp_train(&mlp, train_inputs, train_labels, train_size, batch_size, num_epochs, learning_rate);
    double t_train = get_time_sec() - t_start;

    float loss, accuracy;
    double t_eval_start = get_time_sec();
    mlp_evaluate(&mlp, test_inputs, test_labels, test_size, &loss, &accuracy);
    double t_eval = get_time_sec() - t_eval_start;

    /* Standardized output format — parsed by benchmark.py */
    printf("\n=== Results on %s ===\n", dataset_name);
    printf("Test Loss:     %.4f\n", loss);
    printf("Test Accuracy: %.2f%%\n", accuracy);
    printf("Train time:    %.3f s\n", t_train);
    printf("Eval time:     %.3f s\n", t_eval);
    printf("Throughput:    %.0f samples/s\n", (double)train_size * num_epochs / t_train);

    mlp_free(&mlp);
    free(inputs);
    free(labels);

    return EXIT_SUCCESS;
}
