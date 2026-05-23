#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data.h"

#define LINE_SIZE 20000

Dataset load_mnist_csv(const char *filename, int max_samples) {
    Dataset dataset;
    dataset.size = 0;

    dataset.images = malloc(max_samples * sizeof(double *));
    dataset.labels = malloc(max_samples * sizeof(int));

    if (dataset.images == NULL || dataset.labels == NULL) {
        printf("Error: Could not allocate memory for dataset.\n");
        exit(1);
    }

    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        printf("Error: Could not open file %s\n", filename);
        exit(1);
    }

    char line[LINE_SIZE];

    while (dataset.size < max_samples && fgets(line, LINE_SIZE, file) != NULL) {
        dataset.images[dataset.size] = malloc(INPUT_SIZE * sizeof(double));

        if (dataset.images[dataset.size] == NULL) {
            printf("Error: Could not allocate memory for image.\n");
            exit(1);
        }

        char *token = strtok(line, ",");

        if (token == NULL) {
            continue;
        }

        dataset.labels[dataset.size] = atoi(token);

        for (int i = 0; i < INPUT_SIZE; i++) {
            token = strtok(NULL, ",");

            if (token == NULL) {
                printf("Error: Missing pixel value in row %d\n", dataset.size);
                exit(1);
            }

            dataset.images[dataset.size][i] = atof(token) / 255.0;
        }

        dataset.size++;
    }

    fclose(file);

    return dataset;
}

void free_dataset(Dataset *dataset) {
    for (int i = 0; i < dataset->size; i++) {
        free(dataset->images[i]);
    }

    free(dataset->images);
    free(dataset->labels);

    dataset->images = NULL;
    dataset->labels = NULL;
    dataset->size = 0;
}

void print_sample(Dataset *dataset, int index) {
    if (index < 0 || index >= dataset->size) {
        printf("Error: Invalid sample index.\n");
        return;
    }

    printf("Sample index: %d\n", index);
    printf("Label: %d\n", dataset->labels[index]);

    printf("First 10 normalized pixel values:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", dataset->images[index][i]);
    }

    printf("\n");
}

void print_sample_stats(Dataset *dataset, int index) {
    if (index < 0 || index >= dataset->size) {
        printf("Error: Invalid sample index.\n");
        return;
    }

    int nonzero_count = 0;
    double min_value = dataset->images[index][0];
    double max_value = dataset->images[index][0];

    for (int i = 0; i < INPUT_SIZE; i++) {
        double value = dataset->images[index][i];

        if (value != 0.0) {
            nonzero_count++;
        }

        if (value < min_value) {
            min_value = value;
        }

        if (value > max_value) {
            max_value = value;
        }
    }

    printf("Sample index: %d\n", index);
    printf("Label: %d\n", dataset->labels[index]);
    printf("Nonzero pixels: %d\n", nonzero_count);
    printf("Min pixel value: %.3f\n", min_value);
    printf("Max pixel value: %.3f\n", max_value);
}
