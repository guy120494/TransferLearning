from typing import List

import tensorflow as tf
from tensorflow.python.keras.models import Model

from transfer.models.base_model_builder import build_base_model


def build_fixed_layers_models(model: Model) -> List[Model]:
    models_list: List[Model] = []
    weights = model.get_weights()
    for i in range(1, len(model.layers) + 1):
        if not model.layers[i - 1].trainable_weights:
            continue
        frozen_model = tf.keras.models.clone_model(model)
        frozen_model.set_weights(weights)
        for j in range(i):
            frozen_model.layers[j].trainable = False
        models_list.append(frozen_model)

    return models_list


def main():
    base_model = build_base_model()

    base_model.save('my-model.h5')


if __name__ == '__main__':
    main()
