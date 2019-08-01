import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from transfer.config.config import NUMBER_OF_LABELS
from transfer.datasets.data_builder import get_mnist_data

np.random.seed(123)  # for reproducibility


def build_base_model():
    input_shape, train_data, train_labels, test_data, test_labels = get_mnist_data()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, name='first'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='mid'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='third'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(NUMBER_OF_LABELS, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(),
                  metrics=['accuracy'])

    history = model.fit(x=train_data, y=train_labels, batch_size=128, epochs=15, verbose=1)

    model.evaluate(test_data, test_labels, verbose=0)
    plot_accuracy(history.history['acc'], 'model-accuracy-train')

    return model


def plot_accuracy(accuracy, filename):
    plt.plot(accuracy)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(filename, format='png')


def get_confusion_matrix(model, test_data, test_labels):
    test_predictions = model.predict_classes(test_data)
    array = [0] * NUMBER_OF_LABELS
    for i in range(len(test_labels)):
        for j in range(len(test_labels[i])):
            array[i] += j * test_labels[i, j]
    cm = confusion_matrix(array, test_predictions)
    return cm
