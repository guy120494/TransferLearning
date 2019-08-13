from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model

from transfer.datasets.data_builder import get_mnist_compatible_cifar10
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
    alpha, n_wavelets, errors = wfr.evaluate_smoothness()
    return alpha


def plot_vec(x=0, y=None, title='', xaxis='', yaxis=''):
    if x == 0:
        x = range(1, len(y) + 1)
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.show()


def main():
    model: Model = tf.keras.models.load_model(r'C:\Users\transfer2\PycharmProjects\TransferLearning\base_model.h5')
    _, train_data, train_labels, test_data, test_labels = get_mnist_compatible_cifar10()

    for i in range(1000, 5000 + 1, 1000):
        models = build_fixed_layers_models(model)

        for j in range(len(models)):
            x_train = train_data[0:i]
            y_train = train_labels[0:i]
            x_test = test_data
            y_test = test_labels
            alpha_vec = np.zeros((len(model.layers),))
            models[j].fit(x=x_train, y=y_train, batch_size=64, epochs=30, verbose=2)
            for idx, layer in enumerate(models[j].layers):
                print('Calculating smoothness parameters for layer ' + str(idx) + '.')
                get_layer_output = tf.keras.backend.function([model.layers[0].input],
                                                                     [model.layers[idx].output])
                layer_output = get_layer_output([x_train])[0]
                alpha_vec[idx] = calc_smoothness(layer_output.reshape(-1, layer_output.shape[0]).transpose(),
                                                 y_train)

            score = models[j].evaluate(x=x_test, y=y_test)
            np.save(fr'C:\Users\transfer2\PycharmProjects\TransferLearning\transfer\smoothness_vector_of_model_{j}_and_{i}_train_data.npy', alpha_vec)

            models[j].save(fr'C:\Users\transfer2\PycharmProjects\TransferLearning\transfer\model_{j}_and_{i}_train_data.h5')

            with open(fr'C:\Users\transfer2\PycharmProjects\TransferLearning\transfer\scores_of_model_{j}_and_{i}_train_data.txt', 'w') as f:
                f.write('\t'.join(models[j].metrics_names))
                f.write('\n')
                f.write(f'{score}')


if __name__ == '__main__':
    main()
