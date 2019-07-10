import tensorflow as tf

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
    fac = 0.99 / 255
    (train_data, train_labels), (test_data, test_labels) = dataset.load_data()

    train_data = train_data.reshape(train_data.shape[0], image_shape[0], image_shape[0], image_shape[2])
    test_data = test_data.reshape(test_data.shape[0], image_shape[0], image_shape[0], image_shape[2])
    train_data = train_data * fac + 0.01
    test_data = test_data * fac + 0.01

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=NUMBER_OF_LABELS)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=NUMBER_OF_LABELS)
    return train_data, train_labels, test_data, test_labels
