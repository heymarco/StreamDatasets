from changeds.abstract import RegionalChangeStream
from changeds.datastreams.abrupt import *
from changeds.datastreams.gradual import *


if __name__ == '__main__':
    stream = GradualMNIST()
    print(stream.id())
    print(stream.drift_lengths())
    while stream.has_more_samples():
        x, y, is_change = stream.next_sample()
        if is_change:
            print("Change at index {}".format(stream.sample_idx))

    if isinstance(stream, RegionalChangeStream):
        change_regions = stream.approximate_change_regions()
        print(change_regions)
