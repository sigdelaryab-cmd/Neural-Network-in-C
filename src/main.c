#include <stdio.h>
#include "data.h"

int main() {
    Dataset train_data = load_mnist_csv("data/mnist_train.csv", 1000);
    Dataset test_data = load_mnist_csv("data/mnist_test.csv", 1000);

    printf("Loaded %d training samples.\n", train_data.size);
    printf("Loaded %d testing samples.\n", test_data.size);

    print_sample(&train_data, 0);
    print_sample_stats(&train_data, 0);

    free_dataset(&train_data);
    free_dataset(&test_data);

    return 0;
}
