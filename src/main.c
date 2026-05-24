#include <stdio.h>
#include "data.h"
#include "nn.h"
#include "train.h"

int main() {
    Dataset train_data = load_mnist_csv("data/mnist_train.csv", 1000);
    Dataset test_data = load_mnist_csv("data/mnist_test.csv", 1000);

    printf("Loaded %d training samples.\n", train_data.size);
    printf("Loaded %d testing samples.\n", test_data.size);

    NeuralNetwork net;
    init_network(&net);

    double initial_accuracy = evaluate_accuracy(&net, &test_data);
    printf("Initial test accuracy: %.2f%%\n", initial_accuracy * 100.0);

    int epochs = 5;
    int batch_size = 32;
    double learning_rate = 0.05;

    train_serial(&net, &train_data, epochs, batch_size, learning_rate);

    double final_accuracy = evaluate_accuracy(&net, &test_data);
    printf("Final test accuracy: %.2f%%\n", final_accuracy * 100.0);

    free_dataset(&train_data);
    free_dataset(&test_data);

    return 0;
}
