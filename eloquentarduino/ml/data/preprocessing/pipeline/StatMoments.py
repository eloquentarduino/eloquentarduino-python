import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class StatMoments(BaseStep):
    """
    Compute statistical moments (mean, variance, skew, kurtosis)
    """
    def __init__(self, num_features, moments=4, name='StatMoments'):
        """
        :param num_features: int
        :param moments: int 1 for mean, 2 for variance, 3 for skew, 4 for kurtosis
        """
        assert num_features > 0, 'num_features MUST be greater than 0'
        assert moments > 0, 'moments MUST be greater than 0'
        super().__init__(name)
        self.num_features = num_features
        self.num_moments = moments

    def fit(self, X, y):
        """
        Fit
        """
        self.set_X(X)

        if self.num_features < 1:
            self.num_features = int(self.input_dim * self.num_features)
        
        self.working_dim = self.num_features * self.num_moments

        # nothing to fit
        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Extract moments
        :return: np.ndarray
        """
        assert (self.input_dim % self.num_features) == 0, 'num_features MUST be a divisor of X.shape[1]'

        moments = None

        for k in range(self.num_features):
            samples = X[:, k::self.num_features]

            # mean
            mean = samples.mean(axis=1).reshape((-1, 1))
            moments = mean if moments is None else np.hstack((moments, mean))

            # var
            if self.num_moments > 1:
                var = ((samples - mean) ** 2).mean(axis=1).reshape((-1, 1))
                moments = np.hstack((moments, var))

                # skew
                if self.num_moments > 2:
                    skew = (((samples - mean) ** 3) / (var ** 1.5)).mean(axis=1).reshape((-1, 1))
                    moments = np.hstack((moments, skew))

                    # kurtosis
                    if self.num_moments > 3:
                        kurtosis = (((samples - mean) ** 4) / (var ** 2)).mean(axis=1).reshape((-1, 1))
                        moments = np.hstack((moments, kurtosis))

            # more stats
            #rms = (samples ** 2).mean(axis=1).reshape((-1, 1))
            #waveform_length = np.abs(np.diff(samples, axis=1)).mean(axis=1).reshape((-1, 1))
            #moments = np.hstack((moments, rms, waveform_length))

        moments[np.isnan(moments)] = 0

        return moments, y

    def get_template_data(self):
        """

        """
        return {
            'num_features': self.num_features,
            'num_moments': self.num_moments,
            'num_samples': self.input_dim // self.num_features
        }