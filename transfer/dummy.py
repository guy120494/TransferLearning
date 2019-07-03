# 3. Import libraries and modules
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

NUMBER_OF_LABELS = 10

MNIST_IMAGE_SIZE = 28

np.random.seed(123)  # for reproducibility


def main():
    fac = 0.99 / 255
    mnist = tf.keras.datasets.mnist
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    if tf.keras.backend.image_data_format() == 'channels_first':
        train_data = train_data.reshape(train_data.shape[0], 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
        test_data = test_data.reshape(test_data.shape[0], 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
        input_shape = (1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
    else:
        train_data = train_data.reshape(train_data.shape[0], MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, 1)
        test_data = test_data.reshape(test_data.shape[0], MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, 1)
        input_shape = (MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, 1)

    train_data = train_data * fac + 0.01
    test_data = test_data * fac + 0.01

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=NUMBER_OF_LABELS)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=NUMBER_OF_LABELS)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(NUMBER_OF_LABELS, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD, metrics=['accuracy'])

    history = model.fit(x=train_data, y=train_labels, batch_size=128, epochs=10, verbose=1,
                        validation_split=0.1)

    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
