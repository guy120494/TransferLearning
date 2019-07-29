from typing import List

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
        frozen_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(),
                             metrics=['accuracy'])
        frozen_model.set_weights(weights)
        for j in range(i):
            frozen_model.layers[j].trainable = False
        models_list.append(frozen_model)

    return models_list


def calc_smoothness(x, y):
    wfr = WaveletsForestRegressor(regressor='random_forest', criterion='mse', depth=9, trees=5)
    wfr.fit(x, y)
    alpha, n_wavelets, errors = wfr.evaluate_smoothness(m=100)
    return alpha


def main():
    model: Model = tf.keras.models.load_model('base-model.h5')
    _, train_data, train_labels, test_data, test_labels = get_mnist_compatible_cifar10()

    for i in range(1000, train_data.shape[0], 1000):
        models = build_fixed_layers_models(model)

        for j in range(len(models)):
            x_train = train_data[0:i]
            y_train = train_labels[0:i]
            x_test = test_data[0:i]
            y_test = test_labels[0:i]
            alpha_vec = np.zeros((len(model.layers),))
            models[j].fit(x=x_train, y=y_train, batch_size=128, epochs=15, verbose=1)
            for idx, layer in enumerate(models[j].layers):
                print('Calculating smoothness parameters for layer ' + str(idx) + '.')
                get_layer_output = tf.keras.backend.backend.function([model.layers[0].input],
                                                                     [model.layers[idx].output])
                layer_output = get_layer_output([x_train])[0]
                alpha_vec[idx] = calc_smoothness(layer_output.reshape(-1, layer_output.shape[0]).transpose(),
                                                 y_train)

            scores = models[j].evaluate(x=x_test, y=y_test)


if __name__ == '__main__':
    main()
