import numpy as np
from tensorflow import keras

from changeds.abstract import GradualChangeStream, RegionalChangeStream


_type = "G"


class GradualMNIST(GradualChangeStream, RegionalChangeStream):
    def __init__(self, num_changes: int = 100, drift_length: int = 100, stretch: bool = True, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])
        super(GradualMNIST, self).__init__(X=x, y=y, num_changes=num_changes, drift_length=drift_length,
                                           stretch=stretch, preprocess=preprocess)

    def id(self) -> str:
        return "MNIST"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

    def type(self) -> str:
        return _type
