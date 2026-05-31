#include <stdio.h>
#include <omp.h>

#include "data.h"
#include "nn.h"
#include "train.h"

int main() {
    int train_samples = 60000;
    int test_samples = 10000;
    int num_threads = 4;

    int epochs = 5;
    int batch_size = 32;
    double learning_rate = 0.05;

    Dataset train_data = load_mnist_csv("data/mnist_train.csv", train_samples);
    Dataset test_data = load_mnist_csv("data/mnist_test.csv", test_samples);

    // printf("Sequential Neural Network Training\n");
    printf("OpenMP Neural Network Training\n");
    printf("Threads: %d\n", num_threads);
    printf("Training samples: %d\n", train_data.size);
    printf("Testing samples: %d\n", test_data.size);
    printf("Epochs: %d\n", epochs);
    printf("Batch size: %d\n", batch_size);
    printf("Learning rate: %.4f\n", learning_rate);
    printf("\n");

    NeuralNetwork net;
    init_network(&net);

    double initial_accuracy = evaluate_accuracy(&net, &test_data);
    printf("Initial test accuracy: %.2f%%\n\n", initial_accuracy * 100.0);

    double start_time = omp_get_wtime();

    // train_serial(&net, &train_data, epochs, batch_size, learning_rate);
    train_openmp(&net, &train_data, epochs, batch_size, learning_rate, num_threads);

    
    double end_time = omp_get_wtime();

    double serial_runtime = end_time - start_time;

    double final_accuracy = evaluate_accuracy(&net, &test_data);

    printf("\nSerial training summary:\n");
    printf("Initial test accuracy: %.2f%%\n", initial_accuracy * 100.0);
    printf("Final test accuracy: %.2f%%\n", final_accuracy * 100.0);
    printf("Serial training runtime: %.6f seconds\n", serial_runtime);

    free_dataset(&train_data);
    free_dataset(&test_data);

    return 0;
}
