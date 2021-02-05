import numpy as np
from eloquentarduino.ml.data.loaders import rolling_window
from eloquentarduino.utils import jinja


class RollingWindow:
    MIN = 'min'
    MAX = 'max'
    MEAN = 'mean'
    STD = 'std'
    SKEW = 'skew'
    KURTOSIS = 'kurtosis'

    """
    Process data as a rolling window
    """
    def __init__(self, X, y, window_size, axis=1, num_windows=1, features=None):
        """
        :param X: numpy matrix, each line containing a "continuous motion" of a given class
        :param y:
        """
        assert window_size > 1, "window_size MUST be greater than 1"
        assert num_windows > 0, "num_windows MUST be greater than 0"
        assert axis > 0, "axis MUST be greather than 0"
        assert len(X) == len(y), "X and y MUST have the same length"
        # assert len(y) > 1, "y MUST be of at least 2 elements"
        assert features is None or isinstance(features, list) or isinstance(features, tuple), \
            'features MUST be None or a list'

        if features is None:
            features = [
                RollingWindow.MIN,
                RollingWindow.MAX,
                RollingWindow.MEAN,
                RollingWindow.STD,
                RollingWindow.SKEW,
                RollingWindow.KURTOSIS,
            ]

        self.X = X
        self.y = y
        self.num_windows = num_windows
        self.window_size = window_size
        self.axis = axis
        self.features = set(features)

        if self.has(RollingWindow.SKEW) or self.has(RollingWindow.KURTOSIS):
            self.features.add(RollingWindow.STD)

        if self.has(RollingWindow.STD):
            self.features.add(RollingWindow.MEAN)

        self.Xt, self.Xw, self.yt = self.transform(X, y)

    def has(self, feature):
        """
        Test if a given feature should be calculated
        """
        return feature in self.features

    def transform(self, X, y):
        """
        Transform data
        """
        assert len(X) == len(y), "X and y MUST have the same length"
        #assert len(y) > 1, "y MUST be of at least 2 elements"

        Xt, yt, Xw = None, None, None

        for xi, yi in zip(X, y):
            # algorithm:
            #  1. split into chunks of length window_size * axis
            #  2. reshape to windows_count * window_size * axis
            #  3. for each window, for each axis, compute features
            #  4. chunk into num_windows with overlapping of (num_windows - 1)
            windows_count = len(xi) // (self.window_size * self.axis)
            windows_split = np.arange(1, windows_count + 1) * (self.window_size * self.axis)
            windows = np.split(xi, windows_split)
            windows = np.asarray([w for w in windows if len(w) == (self.window_size * self.axis)])
            windows = windows.reshape((-1, self.window_size, self.axis))
            windows_features = np.asarray([self._extract_features(w) for w in windows])
            rolling_window_features = rolling_window(windows_features, self.num_windows, self.num_windows - 1)
            Xi = np.asarray([self._concatenate(rw) for rw in rolling_window_features])
            yi = np.ones(len(Xi)) * yi

            if Xt is None:
                Xt = Xi
                yt = yi
                Xw = windows
            else:
                Xt = np.vstack((Xt, Xi))
                Xw = np.vstack((Xw, windows))
                yt = np.concatenate((yt, yi))

        return np.nan_to_num(Xt), Xw, yt

    def port(self):
        """
        Port to plain C++
        :return: str plain C++ code
        """
        env = {
            'num_windows': self.num_windows,
            'window_size': self.window_size,
            'axis': self.axis,
            'features': self.features,
            'offset': (self.num_windows - 1) * len(self.features) * self.axis
        }
        return jinja("RollingWindow.jinja", env, pretty=True)

    def _extract_features(self, w):
        """
        Extract features from window
        :param w: window of shape (window_size, axis)
        """
        features = []

        mean = w.mean(axis=0)
        std = np.sqrt(np.sum((w - mean) ** 2, axis=0)) / len(w)

        if self.has('min'):
            features.append(w.min(axis=0))
        if self.has('max'):
            features.append(w.max(axis=0))
        if self.has('mean'):
            features.append(mean)
        if self.has('std'):
            features.append(std)
        if self.has('skew'):
            features.append(np.sum((w - mean) ** 3, axis=0) / std ** 3 / len(w))
        if self.has('kurtosis'):
            features.append(np.sum((w - mean) ** 4, axis=0) / std ** 4 / len(w))

        return np.asarray(features)

    def _concatenate(self, rolling_window):
        """
        Concatenate features of multiple windows
        """
        # each window is arranged as (axis_features, num_axis)
        # reshape in the format (axis0_features, axis1_features, axis2_features, ...)
        return np.concatenate([window.T.flatten() for window in rolling_window])
