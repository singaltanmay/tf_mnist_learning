# https://www.youtube.com/watch?v=cPmjQ9V6Hbk&list=WL&index=10&t=4131s
# conda activate tf2_venv && python3 mnist_example.py

import tensorflow
import matplotlib.pyplot as plt
import numpy as np


def display_some_examples(examples, labels):
    plt.figure(figsize=(10, 10))

    for i in range(25):
        idx = np.random.randint(0, examples.shape[0] - 1)
        img = examples[idx]
        label = labels[idx]
        plt.subplot(5, 5, i + 1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap="gray")

    plt.show()


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)

    display_some_examples(x_train, y_train)
