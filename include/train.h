#ifndef TRAIN_H
#define TRAIN_H

#include "data.h"
#include "nn.h"

void train_serial(NeuralNetwork *net, Dataset *train_data, int epochs, int batch_size, double learning_rate);

void train_openmp(NeuralNetwork *net, Dataset *train_data, int epochs, int batch_size, double learning_rate, int num_threads);

double evaluate_accuracy(NeuralNetwork *net, Dataset *data);

#endif
