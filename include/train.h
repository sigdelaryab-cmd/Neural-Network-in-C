#ifndef TRAIN_H
#define TRAIN_H

#include "data.h"
#include "nn.h"

// Trains the nn sequentially, uses mini-batch gradient descent
void train_serial(NeuralNetwork *net, Dataset *train_data, int epochs, int batch_size, double learning_rate);

// Trains the nn using OpenMP, parallelizes mini batch processing
void train_openmp(NeuralNetwork *net, Dataset *train_data, int epochs, int batch_size, double learning_rate, int num_threads);

// Prediction accuracy of the model on the test dataset
double evaluate_accuracy(NeuralNetwork *net, Dataset *data);

#endif
