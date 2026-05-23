#include "nn.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

void init_network(NeuralNetwork *net) {
    srand(time(NULL));

    // Xavier initialization for W1
    // limit = 1 / sqrt(fan_in) where fan_in = INPUT_SIZE
    double limit1 = 1.0 / sqrt(INPUT_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 2 * limit1 - limit1;
        }
        net->b1[i] = 0.0;
    }

    // Xavier initialization for W2
    // limit = 1 / sqrt(fan_in) where fan_in = HIDDEN_SIZE
    double limit2 = 1.0 / sqrt(HIDDEN_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 2 * limit2 - limit2;
        }
        net->b2[i] = 0.0;
    }
}

void free_network(NeuralNetwork *net) {
    (void)net;
}

void relu(double *z, double *a, int size) {
    // a[i] = max(0, z[i])
    for (int i = 0; i < size; i++) {
        a[i] = (z[i] > 0) ? z[i] : 0.0;
    }
}

double relu_derivative(double z) {
    // ReLU'(z) = 1 if z > 0, else 0
    return (z > 0) ? 1.0 : 0.0;
}

void softmax(double *z, double *a, int size) {
    // Subtract max for numerical stability before exp
    double max_z = z[0];
    for (int i = 1; i < size; i++) {
        if (z[i] > max_z) max_z = z[i];
    }

    // Compute exp(z[i] - max_z) and sum
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        a[i] = exp(z[i] - max_z);
        sum += a[i];
    }

    // Normalize so all probabilities sum to 1
    for (int i = 0; i < size; i++) {
        a[i] /= sum;
    }
}

void forward(NeuralNetwork *net, double input[INPUT_SIZE]) {
    // Layer 1: z1 = W1 * input + b1
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        net->z1[i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->z1[i] += net->W1[i][j] * input[j];
        }
    }

    // Apply ReLU: a1 = ReLU(z1)
    relu(net->z1, net->a1, HIDDEN_SIZE);

    // Layer 2: z2 = W2 * a1 + b2
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        net->z2[i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->z2[i] += net->W2[i][j] * net->a1[j];
        }
    }

    // Apply softmax: a2 = softmax(z2)
    softmax(net->z2, net->a2, OUTPUT_SIZE);
}

double compute_loss(double a2[OUTPUT_SIZE], int label) {
    // Categorical cross-entropy: L = -log(a2[label])
    // Epsilon prevents log(0) if probability rounds to zero
    double epsilon = 1e-7;
    double prob = a2[label];
    if (prob < epsilon) prob = epsilon;
    return -log(prob);
}

int predict(NeuralNetwork *net, double input[INPUT_SIZE]) {
    // Run forward pass then return index of highest probability
    forward(net, input);

    int best = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (net->a2[i] > net->a2[best]) {
            best = i;
        }
    }
    return best;
}

void zero_gradients(Gradients *grad) {
    // Reset all gradient values to 0 before each mini-batch
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        grad->db1[i] = 0.0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            grad->dW1[i][j] = 0.0;
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        grad->db2[i] = 0.0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            grad->dW2[i][j] = 0.0;
        }
    }
}

void compute_gradients(NeuralNetwork *net, double input[INPUT_SIZE],
                       int label, Gradients *grad) {
    double delta2[OUTPUT_SIZE];
    double delta1[HIDDEN_SIZE];

    // Output delta: δ2 = a2 - y_onehot
    // Simplified gradient from softmax + cross-entropy combined
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        delta2[i] = net->a2[i];
        if (i == label) delta2[i] -= 1.0;
    }

    // Accumulate dW2 and db2
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        grad->db2[i] += delta2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            grad->dW2[i][j] += delta2[i] * net->a1[j];
        }
    }

    // Hidden delta: δ1 = (W2^T * δ2) ⊙ ReLU'(z1)
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        delta1[j] = 0.0;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            delta1[j] += net->W2[i][j] * delta2[i];
        }
        delta1[j] *= relu_derivative(net->z1[j]);
    }

    // Accumulate dW1 and db1
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        grad->db1[i] += delta1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            grad->dW1[i][j] += delta1[i] * input[j];
        }
    }
}

void accumulate_gradients(Gradients *total, Gradients *local) {
    // Merge a thread's local gradients into the global total
    // Used during OpenMP parallel mini-batch training
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        total->db1[i] += local->db1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            total->dW1[i][j] += local->dW1[i][j];
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        total->db2[i] += local->db2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            total->dW2[i][j] += local->dW2[i][j];
        }
    }
}

void apply_gradients(NeuralNetwork *net, Gradients *grad,
                     double learning_rate, int batch_size) {
    // Average gradients over batch then subtract from weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        net->b1[i] -= learning_rate * grad->db1[i] / batch_size;
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->W1[i][j] -= learning_rate * grad->dW1[i][j] / batch_size;
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        net->b2[i] -= learning_rate * grad->db2[i] / batch_size;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->W2[i][j] -= learning_rate * grad->dW2[i][j] / batch_size;
        }
    }
}
