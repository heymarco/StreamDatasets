# StreamDatasets

## Usage

Small code example:

    from changeds import SortedMNIST
    
    if __name__ == '__main__':
        stream = SortedMNIST()
        while stream.has_more_samples():
            x, y, is_change = stream.next_sample()
            if is_change:
                print("Change at index {}".format(stream.sample_idx))

## Installation

Atm, you can add the library as a development library via pip:

    python -m pip install -e git+https://github.com/heymarco/StreamDatasets.git#egg=stream-datasets --upgrade