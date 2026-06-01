#include <stdio.h>
#include <omp.h>

#include "data.h"
#include "nn.h"
#include "train.h"

int main() {
    // Model settings
    int train_samples = 60000;
    int test_samples = 10000;
    int num_threads = 8;

    int epochs = 5;
    int batch_size = 32;
    double learning_rate = 0.05;

    // Loading MNIST train and test data
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

    // Initialize the neural network with fixed random weights
    NeuralNetwork net;
    init_network(&net);

    // Check model accuracy before training
    double initial_accuracy = evaluate_accuracy(&net, &test_data);
    printf("Initial test accuracy: %.2f%%\n\n", initial_accuracy * 100.0);

    // Timing only the training section
    double start_time = omp_get_wtime();

    // train_serial(&net, &train_data, epochs, batch_size, learning_rate);
    train_openmp(&net, &train_data, epochs, batch_size, learning_rate, num_threads);

    
    double end_time = omp_get_wtime();

    double runtime = end_time - start_time;

    // check model accuracy after training
    double final_accuracy = evaluate_accuracy(&net, &test_data);

    // printf("\nSerial training summary:\n");
    printf("\nParallel training summary:\n");
    printf("Initial test accuracy: %.2f%%\n", initial_accuracy * 100.0);
    printf("Final test accuracy: %.2f%%\n", final_accuracy * 100.0);
    // printf("Serial training runtime: %.6f seconds\n", runtime);
    printf("Parallel training runtime: %.6f seconds\n", runtime);

    // free dataset memory
    free_dataset(&train_data);
    free_dataset(&test_data);

    return 0;
}
