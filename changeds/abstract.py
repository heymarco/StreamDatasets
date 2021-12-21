from abc import abstractmethod
from skmultiflow.data import DataStream


class ChangeStream(DataStream):
    def next_sample(self, batch_size=1):
        change = self._is_change()
        x, y = super(ChangeStream, self).next_sample(batch_size)
        return x, y, change

    @abstractmethod
    def _is_change(self) -> bool:
        pass