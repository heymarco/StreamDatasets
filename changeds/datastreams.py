import os

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


class PowertoolDataset(ChangeStream):
    def __init__(self, preprocess=None):
        power_tool_split_ratios = [
            5600 / 14200,
            6800 / 10500,
            8700 / 19000,
            7000 / 15000,
            6500 / 18000,
            5500 / 11500,
            3900 / 10500,
            7200 / 12200,
            4400 / 8400,
            5050 / 11000,
            3600 / 7800,
            4900 / 11700,
            2800 / 8100,
            3500 / 7500,
            2400 / 6000
        ]
        dfs = []
        y = []
        change_indices = []
        path = os.path.join("..", "data", "powertool")
        for i, filename in enumerate(sorted(os.listdir(path))):
            filepath = os.path.join(path, filename)
            this_df = pd.read_csv(filepath, sep=";").dropna()
            dfs.append(this_df)
            change_index = int(power_tool_split_ratios[i] * len(this_df))
            change_indices.append(change_index)
            this_y = np.zeros(len(this_df))
            this_y[change_index:] = 1
            y += this_y.tolist()
        x = pd.concat(dfs, ignore_index=True)
        if preprocess:
            x = preprocess(x)
        self._change_points = np.abs(np.diff(y, prepend=y[0])).astype(bool)
        super(PowertoolDataset, self).__init__(data=x, y=np.asarray(y))

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

    def plot_change_region(self, change_idx: int, binary_thresh: float, save: bool, path=None):
        plot_change_region_2d(self, change_idx, binary_thresh, save, path)


if __name__ == '__main__':
    stream = PowertoolDataset()
    while stream.has_more_samples():
        x, y, is_change = stream.next_sample()
        if is_change:
            print("Change at index {}".format(stream.sample_idx))

    if isinstance(stream, RegionalChangeStream):
        change_regions = stream.approximate_change_regions()
        stream.plot_change_region(2, binary_thresh=0.5, save=False)

