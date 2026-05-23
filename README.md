# Parallel Neural Network Training

Our goal is to compare sequential training with OpenMP parallel training using metrics such as runtime, speedup, and accuracy.This project implements a feedforward neural network from scratch in C and parallelizes mini-batch training using OpenMP.

## Project Goal

The goal is to compare sequential neural network training with OpenMP parallel training using:

- Runtime
- Speedup
- Accuracy
- Loss

## Model

The neural network will use:

- Input layer: 784 nodes
- Hidden layer: 64 nodes
- Output layer: 10 nodes
- Hidden activation: ReLU
- Output activation: Softmax
- Loss function: Cross-entropy
- Optimizer: Mini-batch SGD

## Folder Structure
```text
src/        Source files
include/    Header files
data/       MNIST CSV files
results/    Runtime and accuracy results
logs/       Program logs

 
