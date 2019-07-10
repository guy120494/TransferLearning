import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from transfer.config.config import NUMBER_OF_LABELS
from transfer.datasets.data_builder import get_mnist_data

np.random.seed(123)  # for reproducibility


def build_base_model():
    input_shape, train_data, train_labels, test_data, test_labels = get_mnist_data()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(5, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(5, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(NUMBER_OF_LABELS, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(),
                  metrics=['accuracy'])

    history = model.fit(x=train_data, y=train_labels, batch_size=128, epochs=5, verbose=1)

    score = model.evaluate(test_data, test_labels, verbose=0)
    plot_accuracy(history.history['acc'], 'model-accuracy-train')
    # plot_accuracy(score[1], 'model-accuracy-test')
    return model


def plot_accuracy(accuracy, filename):
    plt.plot(accuracy)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(filename, format='png')
