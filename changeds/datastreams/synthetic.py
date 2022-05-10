import numpy as np


from changeds.abstract import RegionalChangeStream, RandomOrderChangeStream, QuantifiesSeverity

_type = "A"


class Hypersphere(RandomOrderChangeStream, RegionalChangeStream, QuantifiesSeverity):
    def __init__(self, num_concepts: int = 100, n_per_concept: int = 2000,
                 dims_drift: int = 50, dims_no_drift: int = 50, preprocess=False, seed=0, random_subspace_size=True):
        self.rng = np.random.default_rng(seed)
        self.dims_drift = dims_drift
        self.dims_no_drift = dims_no_drift
        self.num_concepts = num_concepts
        self.n_per_concept = n_per_concept
        if random_subspace_size:
            total_dims = dims_drift + dims_no_drift
            self.dims_drift = self.rng.integers(low=min(total_dims, 3), high=total_dims)
            self.dims_no_drift = total_dims - self.dims_drift
        data, labels = self._create_data()
        concepts = [i for i in range(num_concepts) for _ in range(n_per_concept)]
        self._change_points = np.diff(concepts, prepend=concepts[0])
        if preprocess:
            data = preprocess(data)
        super(Hypersphere, self).__init__(data=data, y=labels)

    def type(self) -> str:
        return _type

    def id(self) -> str:
        return "HSphere"

    def change_points(self):
        return self._change_points

    def _is_change(self) -> bool:
        return self._change_points[self.sample_idx]

    def _create_hypersphere(self):
        data = self.rng.uniform(-1, 1, size=(self.num_concepts * self.n_per_concept, self.dims_drift))
        return data / np.linalg.norm(data, axis=0)

    def _sample_severity(self):
        return self.rng.integers(low=0, high=100, size=self.num_concepts)

    def _create_data(self):
        y = np.array([i for i in self._sample_severity() for _ in range(self.n_per_concept)])
        data = self._create_hypersphere()
        data = data * np.expand_dims(y, -1)
        uncorrelated = self.rng.normal(scale=0.5 * np.average(y),
                                        size=(self.num_concepts * self.n_per_concept, self.dims_no_drift))
        data = np.concatenate([data, uncorrelated], axis=1)
        return data, y

    def approximate_change_regions(self):
        change_dims = np.arange(self.dims_drift)
        return np.asarray([
            change_dims for cp in self.change_points() if cp
        ])

    def get_severity(self):
        new_label = self.y[self.sample_idx]
        old_label = self.y[self.sample_idx - 2]
        severity = np.abs(new_label - old_label)
        return severity


class Gaussian(RandomOrderChangeStream, RegionalChangeStream, QuantifiesSeverity):
    def __init__(self, num_concepts: int = 100, n_per_concept: int = 2000,
                 dims: int = 20, variance_drift: bool = False,
                 preprocess=False, seed=0):
        self.rng = np.random.default_rng(seed)
        self.num_concepts = num_concepts
        self.n_per_concept = n_per_concept
        self.variance_drift = variance_drift
        self.dims = dims
        data, self.drift_subspace_sizes = self._create_data()
        concepts = [i for i in range(num_concepts) for _ in range(n_per_concept)]
        self._change_points = np.diff(concepts, prepend=concepts[0])
        if preprocess:
            data = preprocess(data)
        super(Gaussian, self).__init__(data=data, y=[])

    def _is_change(self) -> bool:
        return self.change_points()[self.sample_idx]

    def change_points(self):
        return self._change_points

    def _create_data(self):
        if self.variance_drift:
            return self._create_variance_drift()
        else:
            return self._create_mean_drift()

    def _create_mean_drift(self):
        data = self.rng.normal(size=(self.num_concepts, self.n_per_concept, self.dims))
        shift = []
        drift_subspace = []
        for i in range(self.num_concepts):
            if i % 2 == 0:
                shift.append(0)
            else:
                shift.append(self.rng.uniform(low=-1.5, high=1.5))
        for i in range(len(data)):
            subspace = self.rng.uniform(1, self.dims)
            data[i, :, :subspace] += shift[i]
            if i > 0:
                if i % 2 == 0:
                    drift_subspace.append(drift_subspace[-1])
                else:
                    drift_subspace.append(subspace)
        return data.reshape(shape=(self.num_concepts * self.n_per_concept, self.dims)), np.array(drift_subspace)

    def _create_variance_drift(self):
        std = []
        drift_subspace = []
        for i in range(self.num_concepts):
            if i % 2 == 0:
                std.append(np.nan)
            else:
                std.append(self.rng.uniform(low=0.1, high=3))
        std = np.array(std)
        std[std == np.nan] = np.nanmean(std)
        data = self.rng.normal(scale=np.mean(std), size=(self.num_concepts, self.n_per_concept, self.dims))
        for i in range(len(data)):
            subspace = self.rng.uniform(1, self.dims)
            data[i, :, :subspace] = data = self.rng.normal(scale=std[i],
                                                           size=(self.num_concepts, self.n_per_concept, self.dims))
            if i > 0:
                if i % 2 == 0:
                    drift_subspace.append(drift_subspace[-1])
                else:
                    drift_subspace.append(subspace)
        return data, np.array(drift_subspace)

    def approximate_change_regions(self):
        return np.asarray([
            np.arange(subspace) for subspace in self.drift_subspace_sizes
        ])

    def id(self) -> str:
        return "Normal-{}".format("V" if self.variance_drift else "M")

    def type(self) -> str:
        return _type

    def get_severity(self):
        new_label = self.y[self.sample_idx]
        old_label = self.y[self.sample_idx - 2]
        return np.abs(new_label - old_label)

