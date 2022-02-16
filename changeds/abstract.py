from abc import abstractmethod, ABCMeta, ABC
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from skmultiflow.data import DataStream


class ChangeStream(DataStream, metaclass=ABCMeta):
    def next_sample(self, batch_size=1):
        change = self._is_change()
        x, y = super(ChangeStream, self).next_sample(batch_size)
        return x, y, change

    @abstractmethod
    def change_points(self):
        raise NotImplementedError

    @abstractmethod
    def _is_change(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def id(self) -> str:
        raise NotImplementedError


@runtime_checkable
class RegionalChangeStream(Protocol):
    def approximate_change_regions(self):
        change_indices = [i for i, is_change in enumerate(self.change_points()) if is_change == 1]
        change_regions = []
        for i, ci in enumerate(change_indices):
            concept_a = self.data[:ci] if i == 0 else self.data[change_indices[i-1]:ci]
            concept_b = self.data[ci:change_indices[i+1]] if i < len(change_indices)-1 else self.data[ci:len(self.data)]
            mean_concept_a = np.mean(concept_a, axis=0)
            mean_concept_b = np.mean(concept_b, axis=0)
            diff = np.abs(mean_concept_b - mean_concept_a)
            change_regions.append(diff)
        return np.asarray(change_regions)

    @abstractmethod
    def plot_change_region(self, change_idx: int, binary_thresh: float, save: bool, path=None):
        raise NotImplementedError


class ClassificationStream(ChangeStream, ABC):
    def __init__(self, data_path: str, preprocess=None):
        """
        Use this kind of stream to train a classifier with the data and the label
        - can be useful to evaluate the performance of unsupervised change detection for changes in P(X) && P(y|X)
        :param data_path: path to the csv to load
        :param preprocess: preprocessing function
        """
        data = pd.read_csv(data_path)
        y = data["target"].to_numpy()
        x = data.drop("target", axis=1)
        if preprocess:
            x = preprocess(x)
        super(ClassificationStream, self).__init__(data=x, y=y)

    def change_points(self):
        return np.array([np.nan for _ in range(len(self.y))])

    def _is_change(self) -> bool:
        return False


class RandomOrderChangeStream(ChangeStream, ABC):
    @staticmethod
    def create_changes(X, y, num_changes: int, shuffle_within_concept: bool = False):
        sorted_indices = np.argsort(y)
        diffs = np.diff(y[sorted_indices], prepend=0)
        new_concept_indices = [i for i in range(len(diffs)) if diffs[i] == 1]
        concepts = np.split(sorted_indices, new_concept_indices)
        if shuffle_within_concept:
            [np.random.shuffle(concept) for concept in concepts]
        drift_points = []
        x_final = []
        y_final = []
        for change in range(num_changes):
            random_concept = np.random.choice(range(len(concepts)))
            random_concept = concepts[random_concept]
            np.random.shuffle(random_concept)
            x_final.append(X[random_concept])
            y_final.append(y[random_concept])
            if change == num_changes - 1:
                break
            if len(drift_points) > 0:
                drift_points.append(drift_points[-1] + len(random_concept))
            else:
                drift_points.append(len(random_concept))
        x_final = np.concatenate(x_final, axis=0)
        y_final = np.concatenate(y_final, axis=0)
        change_points = np.zeros_like(y_final)
        drift_points = np.asarray(drift_points).astype(int)
        change_points[drift_points] = 1
        return x_final, y_final, change_points

    def change_points(self):
        pass

    def _is_change(self) -> bool:
        pass
