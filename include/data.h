#ifndef DATA_H
#define DATA_H

// Each MNIST Image is 28x28 pixels. 28x28 = 784 input values per image

#define INPUT_SIZE 784

// stores image data, labels, and dataset size
typedef struct {
    double **images;
    int *labels;
    int size;
} Dataset;

// Loads upto max samples rows from an MNIST CSV file
Dataset load_mnist_csv(const char *filename, int max_samples);
// Frees memory allocated for the dataset
void free_dataset(Dataset *dataset);
// prints basic stats for one sample
void print_sample_stats(Dataset *dataset, int index);

#endif
