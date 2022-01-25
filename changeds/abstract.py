from abc import abstractmethod, ABCMeta

import numpy as np
from skmultiflow.data import DataStream


class ChangeStream(DataStream, metaclass=ABCMeta):
    def next_sample(self, batch_size=1):
        change = self._is_change()
        x, y = super(ChangeStream, self).next_sample(batch_size)
        return x, y, change

    @abstractmethod
    def change_points(self):
        pass

    @abstractmethod
    def _is_change(self) -> bool:
        pass


class RegionalChangeStream(ChangeStream, metaclass=ABCMeta):
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
        pass

