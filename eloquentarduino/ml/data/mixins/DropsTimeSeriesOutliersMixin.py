import numpy as np
from banpei import SST


class DropsTimeSeriesOutliersMixin:
    """Implement time-series anomaly detection algorithms

    """
    def drop_singular_spectrum(self, window_size, columns=None, **kwargs):
        """
        :param window_size: int
        :param columns: list or None
        """
        is_outlier = np.zeros(len(self.X))

        for i in range(self.num_features):
            if columns is not None and i not in columns:
                continue

            ts = self.X[:, i]
            model = SST(w=window_size, **kwargs)
            scores = model.detect(ts)
            is_outlier = np.logical_or(is_outlier, scores > scores.mean())

        return self.mask(~is_outlier)
