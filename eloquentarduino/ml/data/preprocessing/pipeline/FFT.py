import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class FFT(BaseStep):
    """
    np.fft.rfft implementation
    """
    def __init__(self, num_features, name='FFT'):
        """
        :param num_features: int how many features are there in the input vector (expected to be flattened)
        """
        assert isinstance(num_features, int) and num_features > 0, 'num_features MUST be positive'

        super().__init__(name)
        self.include_c_library('arduinoFFT.h')
        self.num_features = num_features

    def fit(self, X, y):
        """
        Fit
        """
        fft_samples = X.shape[1] // self.num_features

        assert (fft_samples & (fft_samples - 1) == 0), 'input dimension MUST be a power of 2'

        # nothing to fit
        self.set_X(X)

        self.working_dim = self.input_dim // 2

        return X, y

    def transform(self, X):
        """
        Transform
        """
        fft = None
        for feature_idx in range(self.num_features):
            # arduinoFFT library produces one element less than Numpy (¯\_(ツ)_/¯)
            feature_fft = np.abs(np.fft.rfft(X[:, feature_idx::self.num_features])[:, :-1])
            fft = feature_fft if fft is None else np.hstack((fft, feature_fft))

        return fft

    def get_template_data(self):
        """
        Get template data
        """
        return {
            'num_features': self.num_features,
            'fft_length': int((self.input_dim / self.num_features) // 2)
        }
