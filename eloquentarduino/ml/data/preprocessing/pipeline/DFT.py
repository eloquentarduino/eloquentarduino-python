import numpy as np
from math import pi, cos
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class DFT(BaseStep):
    """
    np.fft.rfft "naive" implementation
    """
    def __init__(self, num_features, lookup_sparsity=0, name='DFT'):
        """
        :param num_features: int how many features are there in the input vector (expected to be flattened)
        """
        assert num_features > 0, 'num_features MUST be positive'
        assert lookup_sparsity >= 0, 'lookup_sparsity MUST be non-negative'

        super().__init__(name)
        self.num_features = num_features
        self.lookup_sparsity = lookup_sparsity

    def get_config(self):
        """
        Get config options
        """
        return {'num_features': self.num_features}

    def fit(self, X, y):
        """
        Fit
        """
        self.set_X(X)

        if self.num_features < 1:
            self.num_features = int(self.input_dim * self.num_features)

        fft_samples = X.shape[1] // self.num_features

        assert (fft_samples & (fft_samples - 1) == 0), 'input dimension MUST be a power of 2'

        self.working_dim = fft_samples // 2 * self.num_features

        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Transform
        """
        fft = None
        for feature_idx in range(self.num_features):
            feature_fft = np.abs(np.fft.rfft(X[:, feature_idx::self.num_features])[:, :-1]) ** 2 # skip sqrt() on MCU
            fft = feature_fft if fft is None else np.hstack((fft, feature_fft))

        return fft, y

    def get_template_data(self):
        """
        Get template data
        """
        return {
            'num_features': self.num_features,
            'num_samples': int(self.input_dim / self.num_features),
            'fft_length': int((self.input_dim / self.num_features) // 2),
            'PI': pi,
            'lookup': [1] + [cos(angle / 360 * 2 * pi) for angle in range(0, 360, self.lookup_sparsity)] + [1] if self.lookup_sparsity > 0 else None
        }
