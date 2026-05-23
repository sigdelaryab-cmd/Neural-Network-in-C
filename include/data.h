#ifndef DATA_H
#define DATA_H

#define INPUT_SIZE 784

typedef struct {
    double **images;
    int *labels;
    int size;
} Dataset;

Dataset load_mnist_csv(const char *filename, int max_samples);
void free_dataset(Dataset *dataset);
void print_sample(Dataset *dataset, int index);
void print_sample_stats(Dataset *dataset, int index);

#endif
