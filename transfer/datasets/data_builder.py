import numpy as np
import tensorflow as tf
from skimage.transform import resize

from transfer.config.config import MNIST_IMAGE_SIZE, CIFAR10_IMAGE_SIZE, NUMBER_OF_LABELS


def get_mnist_data():
    mnist = tf.keras.datasets.mnist
    train_data, train_labels, test_data, test_labels = get_data_from_dataset(mnist, MNIST_IMAGE_SIZE)
    return MNIST_IMAGE_SIZE, train_data, train_labels, test_data, test_labels


def get_cifar10_data():
    cifar_10 = tf.keras.datasets.cifar10
    train_data, train_labels, test_data, test_labels = get_data_from_dataset(cifar_10, CIFAR10_IMAGE_SIZE)
    return CIFAR10_IMAGE_SIZE, train_data, train_labels, test_data, test_labels


def get_data_from_dataset(dataset, image_shape):
    (train_data, train_labels), (test_data, test_labels) = dataset.load_data()

    train_data = train_data.reshape(train_data.shape[0], image_shape[0], image_shape[1], image_shape[2])
    test_data = test_data.reshape(test_data.shape[0], image_shape[0], image_shape[1], image_shape[2])
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_data = train_data / 255
    test_data = test_data / 255

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=NUMBER_OF_LABELS)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=NUMBER_OF_LABELS)
    return train_data, train_labels, test_data, test_labels


def get_mnist_compatible_cifar10():
    input_shape, train_data, train_labels, test_data, test_labels = get_cifar10_data()

    train_data = [rgb2gray(sample) for sample in train_data]
    test_data = [rgb2gray(test_sample) for test_sample in test_data]

    train_data = [resize(image=sample, output_shape=MNIST_IMAGE_SIZE) for sample in train_data]
    test_data = [resize(image=test_sample, output_shape=MNIST_IMAGE_SIZE) for test_sample in test_data]

    return MNIST_IMAGE_SIZE, train_data, train_labels, test_data, test_labels


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
