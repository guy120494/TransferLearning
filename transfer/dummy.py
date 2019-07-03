# 3. Import libraries and modules
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

NUMBER_OF_LABELS = 10

MNIST_IMAGE_SIZE = 28

np.random.seed(123)  # for reproducibility


def main():
    fac = 0.99 / 255
    #takes the mnist dataset
    mnist = tf.keras.datasets.mnist

    #divides data to parts corresponding <->
    #train-
    # data <-> labels
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    #reshapes according to data structure
    if tf.keras.backend.image_data_format() == 'channels_first':
        train_data = train_data.reshape(train_data.shape[0], 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
        test_data = test_data.reshape(test_data.shape[0], 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
        input_shape = (1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
    else:
        train_data = train_data.reshape(train_data.shape[0], MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, 1)
        test_data = test_data.reshape(test_data.shape[0], MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, 1)
        input_shape = (MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, 1)


    #heuristic bias
    train_data = train_data * fac + 0.01
    test_data = test_data * fac + 0.01

    #takes an array (n1,n2,...n_k)
    #so that n_i in {0,1,...t}
    #returns a matrix in the standard basis
    # (,,,3,,,) -> (001000...)
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=NUMBER_OF_LABELS)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=NUMBER_OF_LABELS)


    model = tf.keras.models.Sequential()

    model.add(tf.keras.models.Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape))

    model.add(tf.keras.models.Conv2D(64, (3, 3), activation='relu'))

    # reduce dim, choose best
    model.add(tf.keras.models.MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(0.25))
    model.add(tf.keras.models.Flatten())

    model.add(tf.keras.models.Dense(128, activation='relu'))

    #model.add(Dropout(0.5))
    model.add(tf.keras.models.Dense(NUMBER_OF_LABELS, activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD, metrics=['accuracy'])

    history = model.fit(x=train_data, y=train_labels, batch_size=10, epochs=16, verbose=1)
    result = model.predict(test_data[0:10])
    result = [np.argmax(r) for r in result]
    print("predictions: " + str(result))
    print("actual: " + str([np.argmax(l) for l in test_labels[0:10]]))

    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
