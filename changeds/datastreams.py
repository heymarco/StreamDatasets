from tensorflow import keras
import numpy as np

from abstract import ChangeStream


class SortedMNIST(ChangeStream):
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])
        sorted_indices = np.argsort(y)
        x = x[sorted_indices]
        y = y[sorted_indices]
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(SortedMNIST, self).__init__(data=x, y=y)

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx - 1]


if __name__ == '__main__':
    mnist = SortedMNIST()
    while mnist.has_more_samples():
        x, y, is_change = mnist.next_sample()
        if is_change:
            print("Change at index {}".format(mnist.sample_idx))
