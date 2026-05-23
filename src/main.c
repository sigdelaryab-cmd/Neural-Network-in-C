#include <stdio.h>
#include "data.h"

int main() {
    Dataset train_data = load_mnist_csv("data/mnist_train.csv", 1000);

    printf("Loaded %d training samples.\n", train_data.size);

    print_sample(&train_data, 0);

    free_dataset(&train_data);

    return 0;
}
