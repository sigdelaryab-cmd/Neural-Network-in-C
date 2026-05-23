#ifndef NN_H
#define NN_H

#include "data.h"

#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 10

typedef struct {
    double W1[HIDDEN_SIZE][INPUT_SIZE];
    double b1[HIDDEN_SIZE];

    double W2[OUTPUT_SIZE][HIDDEN_SIZE];
    double b2[OUTPUT_SIZE];

    double z1[HIDDEN_SIZE];
    double a1[HIDDEN_SIZE];

    double z2[OUTPUT_SIZE];
    double a2[OUTPUT_SIZE];
} NeuralNetwork;

void init_network(NeuralNetwork *net);
void forward(NeuralNetwork *net, double input[INPUT_SIZE]);
int predict(NeuralNetwork *net, double input[INPUT_SIZE]);
double compute_loss(NeuralNetwork *net, int label);
void backward(NeuralNetwork *net, double input[INPUT_SIZE], int label, double learning_rate);

#endif
