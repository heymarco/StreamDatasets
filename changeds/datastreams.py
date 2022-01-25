import pandas as pd
from tensorflow import keras
import numpy as np
from skmultiflow.data import HyperplaneGenerator

from changeds.abstract import ChangeStream, RegionalChangeStream
from changeds.helper import plot_change_region_2d


class SortedMNIST(RegionalChangeStream):
    def __init__(self, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])
        sorted_indices = np.argsort(y)
        x = x[sorted_indices]
        y = y[sorted_indices]
        if preprocess:
            x = preprocess(x)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(SortedMNIST, self).__init__(data=x, y=y)

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

    def plot_change_region(self, change_idx: int, binary_thresh: float, save: bool, path=None):
        plot_change_region_2d(self, change_idx, binary_thresh, save, path)


class SortedFashionMNIST(RegionalChangeStream):
    def __init__(self, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train, y_test])
        sorted_indices = np.argsort(y)
        x = x[sorted_indices]
        y = y[sorted_indices]
        if preprocess:
            x = preprocess(x)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(SortedFashionMNIST, self).__init__(data=x, y=y)

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

    def plot_change_region(self, change_idx: int, binary_thresh: float, save: bool, path=None):
        plot_change_region_2d(self, change_idx, binary_thresh, save, path)


class SortedCIFAR10(RegionalChangeStream):
    def __init__(self, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.dot([0.299, 0.587, 0.114])
        x_test = x_test.dot([0.299, 0.587, 0.114])
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train.reshape(-1), y_test.reshape(-1)])
        sorted_indices = np.argsort(y)
        x = x[sorted_indices]
        y = y[sorted_indices]

        if preprocess:
            x = preprocess(x)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(SortedCIFAR10, self).__init__(data=x, y=y)

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

    def plot_change_region(self, change_idx: int, binary_thresh: float, save: bool, path=None):
        plot_change_region_2d(self, change_idx, binary_thresh, save, path)


class SortedCIFAR100(RegionalChangeStream):
    def __init__(self, preprocess=None):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        x_train = x_train.dot([0.299, 0.587, 0.114])
        x_test = x_test.dot([0.299, 0.587, 0.114])
        x_train = np.reshape(x_train, newshape=(len(x_train), x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, newshape=(len(x_test), x_test.shape[1] * x_test.shape[2]))
        x = np.vstack([x_train, x_test])
        y = np.hstack([y_train.reshape(-1), y_test.reshape(-1)])
        sorted_indices = np.argsort(y)
        x = x[sorted_indices]
        y = y[sorted_indices]

        if preprocess:
            x = preprocess(x)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(SortedCIFAR100, self).__init__(data=x, y=y)

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

    def plot_change_region(self, change_idx: int, binary_thresh: float, save: bool, path=None):
        plot_change_region_2d(self, change_idx, binary_thresh, save, path)


class Hyperplane(ChangeStream):

    def __init__(self, preprocess=None, n_drifts: int = 9, n_concept: int = 1000,
                 n_features: int = 100, n_drift_features: int = 10, mag_change: float = 0.5):
        self.n_concept = n_concept
        self.n_drifts = n_drifts
        drift_indices = [(i + 1) * n_concept for i in range(n_drifts)]
        self._change_points = np.zeros(shape=(n_drifts + 1) * n_concept, dtype=int)
        self._change_points[drift_indices] = 1
        x = []
        y = []
        for i in range(n_drifts + 1):
            generator = HyperplaneGenerator(n_features=n_features,
                                            n_drift_features=n_drift_features,
                                            mag_change=mag_change)
            this_concept = [generator.next_sample() for _ in range(n_concept)]
            data = [tpl[0][0] for tpl in this_concept]
            labels = [tpl[1][0] for tpl in this_concept]
            x += data
            y += labels
        x = np.asarray(x)
        if preprocess:
            x = preprocess(x)
        super(Hyperplane, self).__init__(data=x, y=np.asarray(y))

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


if __name__ == '__main__':
    stream = Hyperplane()
    while stream.has_more_samples():
        x, y, is_change = stream.next_sample()
        if is_change:
            print("Change at index {}".format(stream.sample_idx))

    if isinstance(stream, RegionalChangeStream):
        change_regions = stream.approximate_change_regions()
        stream.plot_change_region(2, binary_thresh=0.5, save=False)

