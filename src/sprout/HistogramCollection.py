from cEEC_utils.Histogram1D import Histogram1D
import itertools
import numpy as np

class HistogramCollection(object):
    def __init__(self, **kwargs):
        for values in kwargs.values():
            assert isinstance(values, list)
        self.kwargs = kwargs
        self.keys = tuple(kwargs.keys())
        # self.maxpoint = max_order
        self._generate_histograms()

    def _generate_histograms(self):
        self.IDs = {}
        for values in itertools.product(*self.kwargs.values()):
            self._generate_histogram(values)

    def _generate_histogram(self, values):
        key = self._sorted_key_to_string(values)
        histogram = Histogram1D(key, [0, 1], [0, 1], np.array([-0.5, 0.5, 1.5]), [0, 1], [0, 1], "log")
        self.IDs[key] = histogram

    def _sorted_key_to_string(self, values):
        return "_".join([f"{key}_{value}" for key, value in zip(self.keys, values)])
    def _unsorted_key_to_string(self, kwargs):
        return "_".join([f"{key}_{kwargs[key]}" for key in self.keys])

    def get(self, **kwargs):
        self._verify_keys(kwargs.keys())
        keystr = self._unsorted_key_to_string(kwargs)
        try:
            return self.IDs[keystr]
        except KeyError as e:
            print(f"Your keystring {keystr} was not found.")

    def _verify_keys(self, keys):
        pass