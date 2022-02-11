import os

from tensorflow import keras
import numpy as np

from skmultiflow.data import led_generator, random_rbf_generator

from changeds.abstract import ChangeStream, RegionalChangeStream, ClassificationStream
from changeds.helper import plot_change_region_2d, preprocess_hipe


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


class HIPE(ChangeStream):
    def __init__(self, preprocess=None):
        x = preprocess_hipe()
        y = np.zeros(shape=len(x))
        if preprocess:
            x = preprocess(x)
        self._change_points = y
        super(HIPE, self).__init__(data=x, y=y)

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]


class LED(ChangeStream):
    def __init__(self, n_per_concept: int = 10000, n_drifts: int = 100, has_noise=True, preprocess=None):
        """
        Creates a sudden, but
        :param n_per_concept:
        :param n_drifts:
        :param has_noise:
        :param preprocess:
        """
        random_state = 0
        x = []
        for i in range(n_drifts):
            x.append(led_generator.LEDGenerator(random_state=random_state, has_noise=has_noise,
                                                noise_percentage=(i + 1) / n_drifts if i % 2 == 1 else 0
                                                ).next_sample(n_per_concept)[0])
        y = [i for i in range(n_drifts) for _ in range(n_per_concept)]
        x = np.concatenate(x, axis=0)
        if preprocess:
            x = preprocess(x)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(LED, self).__init__(data=x, y=np.array(y))

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self.change_points()[self.sample_idx]


class RBF(ChangeStream):
    def __init__(self, n_per_concept: int = 10000, n_drifts: int = 100, dims: int = 100,
                 n_centroids: int = 10, preprocess=None):
        sample_random_state = 0
        x = []
        for i in range(n_drifts):
            model_random_state = i
            x.append(random_rbf_generator.RandomRBFGenerator(model_random_state=model_random_state,
                                                             sample_random_state=sample_random_state, n_features=dims,
                                                             n_centroids=n_centroids).next_sample(n_per_concept)[0])
        y = [i for i in range(n_drifts) for _ in range(n_per_concept)]
        x = np.concatenate(x, axis=0)
        if preprocess:
            x = preprocess(x)
        self._change_points = np.diff(y, prepend=y[0]).astype(bool)
        super(RBF, self).__init__(data=x, y=np.array(y))

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self.change_points()[self.sample_idx]


class ArtificialStream(ClassificationStream):
    def __init__(self, filename: str):
        path, _ = os.path.split(__file__)
        path = os.path.join(path, "..", "concept-drift-datasets-scikit-multiflow", "artificial")
        file_path = os.path.join(path, filename)
        assert os.path.exists(file_path), "The requested file does not exist in {}".format(file_path)
        super(ArtificialStream, self).__init__(data_path=file_path)


class RealWorldStream(ClassificationStream):
    def __init__(self, filename: str):
        path, _ = os.path.split(__file__)
        path = os.path.join(path, "..", "concept-drift-datasets-scikit-multiflow", "real-world")
        file_path = os.path.join(path, filename)
        assert os.path.exists(file_path), "The requested file does not exist in {}".format(file_path)
        super(RealWorldStream, self).__init__(data_path=file_path)


if __name__ == '__main__':
    stream = RBF()
    while stream.has_more_samples():
        x, y, is_change = stream.next_sample()
        if is_change:
            print("Change at index {}".format(stream.sample_idx))

    if isinstance(stream, RegionalChangeStream):
        change_regions = stream.approximate_change_regions()
        stream.plot_change_region(2, binary_thresh=0.5, save=False)
