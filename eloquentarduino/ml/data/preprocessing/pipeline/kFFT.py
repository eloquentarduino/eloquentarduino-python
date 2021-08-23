import numpy as np
from math import pi
from numpy.fft import rfft
from collections import Counter
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class kFFT(BaseStep):
    def __init__(self, k, num_features, name='kFFT'):
        """
        :param k: int number of components to keep
        :param num_features: int
        """
        assert k > 0, 'k MUST be a positive number'

        super().__init__(name)
        self.k = k
        self.num_features = num_features
        self.idx = None

    def fit(self, X, y=None):
        """
        Fit
        (find the n top components of the FFT)
        """
        self.set_X(X)
        self.working_dim = self.k * self.num_features

        candidates = []

        for feature_idx in range(self.num_features):
            n = max(10, self.k * 2)
            ffti = np.abs(np.fft.rfft(X[:, feature_idx::self.num_features]))
            idx = (-ffti).argsort()[:, :n].flatten().tolist()
            most_common = [j for j, count in Counter(idx).most_common(n)]
            candidates += most_common

        self.idx = np.asarray([idx for idx, count in Counter(candidates).most_common(self.k)])

        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Transform
        """
        fft = np.empty((len(X), self.k * self.num_features), dtype=X.dtype)

        for i in range(self.num_features):
            ffti = np.abs(np.fft.rfft(X[:, i::self.num_features]))[:, self.idx]
            fft[:, i::self.num_features] = ffti

        return fft, y

    def get_config(self):
        """
        
        """
        return {
            'k': self.k,
            'num_features': self.num_features
        }

    def get_template_data(self):
        """

        """
        return {
            'k': self.k,
            'num_features': self.num_features,
            'output_dim': self.num_features * self.k,
            'idx': self.idx,
            'harmonics': self.idx * 2 * pi / (self.input_dim // self.num_features),
        }
