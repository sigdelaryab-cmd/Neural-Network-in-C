#include <stdio.h>
#include "train.h"
#include <omp.h>


double evaluate_accuracy(NeuralNetwork *net, Dataset *data) {
    int correct = 0;

    for (int i = 0; i < data->size; i++) {
        int prediction = predict(net, data->images[i]);

        if (prediction == data->labels[i]) {
            correct++;
        }
    }

    return (double)correct / data->size;
}

void train_serial(NeuralNetwork *net, Dataset *train_data,
                  int epochs, int batch_size, double learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        int samples_seen = 0;

        for (int start = 0; start < train_data->size; start += batch_size) {
            int current_batch_size = batch_size;

            if (start + current_batch_size > train_data->size) {
                current_batch_size = train_data->size - start;
            }

            Gradients total_grad;
            zero_gradients(&total_grad);

            for (int i = start; i < start + current_batch_size; i++) {
                forward(net, train_data->images[i]);

                total_loss += compute_loss(net->a2, train_data->labels[i]);

                compute_gradients(net,
                                  train_data->images[i],
                                  train_data->labels[i],
                                  &total_grad);

                samples_seen++;
            }

            apply_gradients(net, &total_grad, learning_rate, current_batch_size);
        }

        double average_loss = total_loss / samples_seen;
        double train_accuracy = evaluate_accuracy(net, train_data);

        printf("Epoch %d/%d - Loss: %.6f - Training Accuracy: %.2f%%\n",
               epoch + 1,
               epochs,
               average_loss,
               train_accuracy * 100.0);
    }
}

void train_openmp(NeuralNetwork *net, Dataset *train_data,
                  int epochs, int batch_size,
                  double learning_rate, int num_threads) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        int samples_seen = 0;

        for (int start = 0; start < train_data->size; start += batch_size) {
            int current_batch_size = batch_size;

            if (start + current_batch_size > train_data->size) {
                current_batch_size = train_data->size - start;
            }

            Gradients total_grad;
            zero_gradients(&total_grad);

            double batch_loss = 0.0;

            #pragma omp parallel num_threads(num_threads)
            {
                ForwardCache cache;
                Gradients local_grad;
                zero_gradients(&local_grad);

                double local_loss = 0.0;

                #pragma omp for
                for (int i = start; i < start + current_batch_size; i++) {
                    forward_cached(net, train_data->images[i], &cache);

                    local_loss += compute_loss(cache.a2, train_data->labels[i]);

                    compute_gradients_cached(net,
                                             &cache,
                                             train_data->images[i],
                                             train_data->labels[i],
                                             &local_grad);
                }

                #pragma omp critical
                {
                    accumulate_gradients(&total_grad, &local_grad);
                    batch_loss += local_loss;
                }
            }

            total_loss += batch_loss;
            samples_seen += current_batch_size;

            apply_gradients(net, &total_grad, learning_rate, current_batch_size);
        }

        double average_loss = total_loss / samples_seen;
        double train_accuracy = evaluate_accuracy(net, train_data);

        printf("Epoch %d/%d - Loss: %.6f - Training Accuracy: %.2f%%\n",
               epoch + 1,
               epochs,
               average_loss,
               train_accuracy * 100.0);
    }
}
