"""Train and save the CNN used by the realtime MNIST application."""
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

SEED = 42
IMAGE_SIZE = 28
CHANNELS = 1
EPOCHS = 20
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
MODEL_PATH = Path(__file__).resolve().parent / "mnist_cnn.keras"


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    x_test = x_test.reshape(x_test.shape[0], IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

    # Architecture intentionally matches the model specified for this project.
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)))
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

    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
    )
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Test loss :", loss)
    print("Test accuracy :", accuracy)

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
