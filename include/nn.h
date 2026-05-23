#ifndef NN_H
#define NN_H

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

typedef struct {
  // Layer 1 : Input layer to hidden layer

  double W1 [HIDDEN_SIZE][INPUT_SIZE]; // matrix for the nodes going from the input layer to the hidden layer
  double b1 [HIDDEN_SIZ]; // matrix for the bias values to be added to each node in the hidden layer

  // Layer 2 : Hidden layer to output layer
  double W2 [OUTPUT_SIZE][HIDDEN_SIZE]; // matrix for the nodes going from the hidden layer to the output layer
  double b2 [OUTPUT_SIZE];  // matrix for the bias values to be added to each node in the output layer

  // Forward Pass
  double z1[HIDDEN_SIZE]; // pre-activation hidden layer
  double a1[HIDDEN_SIZE]; // post-activatin hidden layer
  double z2[OUTPUT_SIZE]; // pre-activation output layer
  double a2[OUTPUT_SIZE]; // softmax output
} NeuralNetwork;

typedef struct {
  double dW1[HIDDEN_SIZE][INPUT_SIZE];
  double db1[HIDDEN_SIZE];
  double dW2[OUTPUT_SIZE][HIDDEN_SIZE];
  double db2[OUTPUT_SIZE];
} Gradients;

// Initialization
void init_network(NeuralNetwork *net);
void free_network(NeuralNetwork *net);

// Forward pass
void forward(NeuralNetwork *net, double input[INPUT_SIZE]);

// Activation functions
void relu(double *z, double *a, int size);
double relu_derivative(double z);
void softmax(double *z, double *a, int size);

// Loss (Cross-Entropy Loss)
double compute_loss(double a2[OUTPUT_SIZE], int label);

// Prediction
int predict(NeuralNetwork *net, double input[INPUT_SIZE]);

// Backward pass
void backward(NeuralNetwork *net, double input[INPUT_SIZE], int label, 
              double learning_rate);

// Gradient accumulation (for mini-batch training)
void accumulate_gradients(Gradients *total, Gradients *local);
void zero_gradients(Gradients *grad);
void apply_gradients(NeuralNetwork *net, Gradients *grad, double learning_rate, int batch_size);

#endif
