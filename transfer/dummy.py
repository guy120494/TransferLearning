from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model

from transfer.datasets.data_builder import get_mnist_compatible_cifar10, get_mnist_data
from transfer.smoothness.random_forest import WaveletsForestRegressor


def build_fixed_layers_models(model: Model) -> List[Model]:
    models_list: List[Model] = []
    weights = model.get_weights()
    for i in range(1, len(model.layers) + 1):
        if not model.layers[i - 1].trainable_weights:
            continue
        frozen_model = tf.keras.models.clone_model(model)
        for j in range(i):
            frozen_model.layers[j].trainable = False

        frozen_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(),
                             metrics=['accuracy'])
        frozen_model.set_weights(weights)
        models_list.append(frozen_model)

    return models_list


def calc_smoothness(x, y):
    wfr = WaveletsForestRegressor(regressor='random_forest', criterion='mse', depth=9, trees=5)
    wfr.fit(x, y)
    alpha, n_wavelets, errors = wfr.evaluate_smoothness(m=100)
    return alpha


def plot_vec(x=0, y=None, title='', xaxis='', yaxis=''):
    if x == 0:
        x = range(1, len(y) + 1)
    a = plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    # plt.show()
    a.savefig(f'{title}')


def main():
    # model: Model = tf.keras.models.load_model(r'../base_model.h5')
    _, train_data, train_labels, test_data, test_labels = get_mnist_compatible_cifar10()
    trains = [5000, 10000, 15000, 25000]
    for i in trains:
        model: Model = tf.keras.models.load_model(r'base-model-thin-better.h5')
        models = build_fixed_layers_models(model)

        for j in range(len(models)):
            x_train = train_data[0:i]
            y_train = train_labels[0:i]
            x_test = test_data
            y_test = test_labels
            alpha_vec = np.zeros((len(model.layers),))
            current_model = models[j]
            current_model.fit(x=x_train, y=y_train, batch_size=64, epochs=100, verbose=2)
            for idx, layer in enumerate(models[j].layers):
                print('Calculating smoothness parameters for layer ' + str(idx) + '.')
                get_layer_output = tf.keras.backend.function([current_model.layers[0].input],
                                                             [current_model.layers[idx].output])
                layer_output = get_layer_output([train_data])[0]
                # alpha_vec[idx] = calc_smoothness(layer_output.reshape(-1, layer_output.shape[0]).transpose(),
                #                                  train_labels[0:20000])
                alpha_vec[idx] = calc_smoothness(layer_output.reshape(layer_output.shape[0],
                                                                      np.prod(layer_output.shape[1:])),
                                                 train_labels)

            score = current_model.evaluate(x=x_test, y=y_test)
            np.save(f'smoothness_vector_of_model_{j}_and_{i}_train_data_new_reshape_and_base_better.npy', alpha_vec)
            plot_vec(y=alpha_vec, title=f'Graph of model {j} and {i} train and base better')

            models[j].save(f'model_{j}_and_{i}_train_data_new_reshape_and_base_better.h5')

            with open(f'scores_of_model_{j}_and_{i}_train_data_new_reshape_and_base_better.txt', 'w') as f:
                f.write('\t'.join(models[j].metrics_names))
                f.write('\n')
                f.write(f'{score}')


def base_model_smoothness():
    model: Model = tf.keras.models.load_model(r'base-model-thin-better.h5')
    # model: Model = tf.keras.models.load_model(r'model_3_and_20000_train_data.h5')
    _, train_data, train_labels, test_data, test_labels = get_mnist_data()
    # train_data = train_data[0:20000]
    # train_labels = train_labels[0:20000]
    alpha_vec = np.zeros((len(model.layers),))
    for idx, layer in enumerate(model.layers):
        print('Calculating smoothness parameters for layer ' + str(idx) + '.')
        get_layer_output = tf.keras.backend.function([model.layers[0].input],
                                                     [model.layers[idx].output])
        layer_output = get_layer_output([train_data])[0]
        # alpha_vec[idx] = calc_smoothness(layer_output.reshape(-1, layer_output.shape[0]).transpose(),
        #                                  train_labels)
        alpha_vec[idx] = calc_smoothness(layer_output.reshape(layer_output.shape[0], np.prod(layer_output.shape[1:])),
                                         train_labels)
    score = model.evaluate(x=test_data, y=test_labels)
    np.save(f'smoothness_vector_of_base_thin_better_model.npy', alpha_vec)
    plot_vec(y=alpha_vec, title=f'Graph of base thin better model')
    with open(f'scores_of_base_thin_model_new_reshape.txt', 'w') as f:
        f.write('\t'.join(model.metrics_names))
        f.write('\n')
        f.write(f'{score}')


def make_accuracy_and_loss_graph_for_models():
    x = [k for k in range(1, 5 + 1)]
    trains = [100, 200, 300, 500, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 15000]
    colors = ['r', 'g', 'b', 'c', 'm', 'k', 'y', 'indigo', 'firebrick', 'lightgreen', 'peru', 'gold']
    accuracies = []
    losses = []
    for j in trains:
        accuracy = []
        loss = []
        for i in range(5):
            with open(fr'DanielDir/scores_of_model_{i}_and_{j}_train_data_new_reshape_and_base_better.txt', 'r') as f:
                lines = f.readlines()
                scores = lines[1].replace('[', '').replace(']', '').split(',')
                scores = [float(s) for s in scores]
                loss.append(scores[0])
                accuracy.append(scores[1])

        accuracies.append(accuracy)
        losses.append(loss)

    plt.figure()
    plt.xlabel('number of frozen layers')
    plt.ylabel('accuracy')
    plt.title(f'accuracy over models')
    for i in range(len(accuracies)):
        plt.plot(x, accuracies[i], color=colors[i], label=f'{trains[i]} train samples')
    plt.legend(loc='lower left')
    plt.savefig(f'DanielDir/accuracy over models')
    plt.close()

    plt.figure()
    plt.xlabel('number of frozen layers')
    plt.ylabel('loss')
    plt.title(f'loss over models')
    for i in range(len(accuracies)):
        plt.plot(x, losses[i], color=colors[i], label=f'{trains[i]} train samples')
    plt.legend(loc='upper left')
    plt.savefig(f'DanielDir/loss over models')
    plt.close()


if __name__ == '__main__':
    # base_model_smoothness()
    # main()
    make_accuracy_and_loss_graph_for_models()
