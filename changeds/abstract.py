from abc import ABC, abstractmethod
from skmultiflow.data import DataStream


class ChangeStream(DataStream):
    def next_sample(self, batch_size=1):
        x, y = super(ChangeStream, self).next_sample(batch_size)
        change = self._is_change()
        return x, y, change

    @abstractmethod
    def _is_change(self) -> bool:
        pass