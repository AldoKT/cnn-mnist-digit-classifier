"""Train and save the CNN used by the realtime MNIST application."""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

    # Architecture intentionally matches the model specified for this project.
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, epochs=20, validation_split=0.1)
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Accuracy :", accuracy)

    model.save("mnist_cnn.keras")
    print("Model saved to mnist_cnn.keras")


if __name__ == "__main__":
    main()
