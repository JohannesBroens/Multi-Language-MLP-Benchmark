#include "data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LINE_LENGTH 1024

/*
 * Generate synthetic 2D circle classification data.
 * Points in [-1, 1]^2; label 1 if inside circle of radius 0.5, else 0.
 */
int load_generated_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size, int num_samples_requested) {
    *input_size = 2;
    *output_size = 2;   /* binary classification */
    *num_samples = (num_samples_requested > 0) ? num_samples_requested : 1000;

    *inputs = (float *)malloc((*num_samples) * (*input_size) * sizeof(float));
    *labels = (int *)malloc((*num_samples) * sizeof(int));

    if (!(*inputs) || !(*labels)) {
        fprintf(stderr, "Memory allocation failed for inputs or labels.\n");
        return -1;
    }

    srand((unsigned int)time(NULL));
    for (int i = 0; i < *num_samples; ++i) {
        float x = ((float)rand() / RAND_MAX) * 2 - 1;
        float y = ((float)rand() / RAND_MAX) * 2 - 1;
        (*inputs)[i * (*input_size)] = x;
        (*inputs)[i * (*input_size) + 1] = y;
        (*labels)[i] = (x * x + y * y < 0.25f) ? 1 : 0;
    }

    return 0;
}

/*
 * Load Iris dataset from preprocessed file.
 * Format: comma-separated, 4 float features then float label per line.
 * File: data/iris_processed.txt (150 samples, 3 classes)
 */
int load_iris_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size) {
    const char *filename = "data/iris_processed.txt";
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open Iris dataset file");
        return -1;
    }

    *input_size = 4;
    *output_size = 3;

    /* Count lines to determine sample count */
    *num_samples = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        (*num_samples)++;
    }
    rewind(file);

    *inputs = (float *)malloc((*num_samples) * (*input_size) * sizeof(float));
    *labels = (int *)malloc((*num_samples) * sizeof(int));

    if (!(*inputs) || !(*labels)) {
        fprintf(stderr, "Memory allocation failed for inputs or labels.\n");
        fclose(file);
        return -1;
    }

    /* Labels are stored as floats in the preprocessed file; round to int */
    for (int i = 0; i < *num_samples; ++i) {
        for (int j = 0; j < *input_size; ++j) {
            fscanf(file, "%f,", &(*inputs)[i * (*input_size) + j]);
        }
        float label_f;
        fscanf(file, "%f", &label_f);
        (*labels)[i] = (int)(label_f + 0.5f);
    }

    fclose(file);
    return 0;
}

/*
 * Load UCI Wine Quality dataset.
 * Format: semicolon-delimited CSV with header row.
 * 11 float features; last column is quality score (0-10) used as label.
 * File: data/winequality-red.csv or data/winequality-white.csv
 */
int load_wine_quality_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size, const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open Wine Quality dataset file");
        return -1;
    }

    *input_size = 11;
    *output_size = 11;  /* quality scores 0-10 */

    /* Skip header line */
    char line[MAX_LINE_LENGTH];
    fgets(line, sizeof(line), file);

    *num_samples = 0;
    while (fgets(line, sizeof(line), file)) {
        (*num_samples)++;
    }
    rewind(file);
    fgets(line, sizeof(line), file);  /* skip header again after rewind */

    *inputs = (float *)malloc((*num_samples) * (*input_size) * sizeof(float));
    *labels = (int *)malloc((*num_samples) * sizeof(int));

    if (!(*inputs) || !(*labels)) {
        fprintf(stderr, "Memory allocation failed for inputs or labels.\n");
        fclose(file);
        return -1;
    }

    for (int i = 0; i < *num_samples; ++i) {
        for (int j = 0; j < *input_size; ++j) {
            fscanf(file, "%f;", &(*inputs)[i * (*input_size) + j]);
        }
        fscanf(file, "%d", &(*labels)[i]);
    }

    fclose(file);
    return 0;
}

/*
 * Load Wisconsin Breast Cancer Diagnostic dataset.
 * Format: id,diagnosis(M/B),30 float features — comma-separated, no header.
 * M (malignant) = 1, B (benign) = 0.
 * File: data/wdbc.data (569 samples, 2 classes)
 */
int load_breast_cancer_data(float **inputs, int **labels, int *num_samples, int *input_size, int *output_size) {
    const char *filename = "data/wdbc.data";
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open Breast Cancer dataset file");
        return -1;
    }

    *input_size = 30;
    *output_size = 2;   /* malignant vs benign */

    *num_samples = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        (*num_samples)++;
    }
    rewind(file);

    *inputs = (float *)malloc((*num_samples) * (*input_size) * sizeof(float));
    *labels = (int *)malloc((*num_samples) * sizeof(int));

    if (!(*inputs) || !(*labels)) {
        fprintf(stderr, "Memory allocation failed for inputs or labels.\n");
        fclose(file);
        return -1;
    }

    /* Parse: id (discarded), diagnosis char, then 30 features */
    for (int i = 0; i < *num_samples; ++i) {
        int id;
        char diagnosis;
        fscanf(file, "%d,%c,", &id, &diagnosis);
        (*labels)[i] = (diagnosis == 'M') ? 1 : 0;

        for (int j = 0; j < *input_size; ++j) {
            fscanf(file, "%f,", &(*inputs)[i * (*input_size) + j]);
        }
    }

    fclose(file);
    return 0;
}
