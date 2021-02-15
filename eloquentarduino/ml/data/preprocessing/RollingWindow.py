import numpy as np
from eloquentarduino.utils import jinja


class RollingWindow:
    """
    Process data as a rolling window
    """
    def __init__(self, depth, shift=1):
        """
        :param depth: how many samples will form a single window
        :param shift: how many samples to skip between windows
        """
        assert depth > 1, "depth MUST be greater than 1"
        assert shift > 0, "shift MUST be greater than 0"
        self.depth = int(depth)
        self.shift = int(shift)
        self.input_dim = None

    def fit(self, X):
        """
        "Fit" the rolling window
        """
        self.transform(X)

    def transform(self, X, flatten=True):
        """
        Transform data
        :param X: input data
        :return: np.ndarray transformed input
        """
        self.input_dim = X.shape[1]
        w = np.arange(self.depth)
        t = np.arange(len(X) - self.depth + 1)
        idx = (w + t.reshape((-1, 1)))
        idx = idx[::self.shift]
        return X[idx].reshape((-1, self.input_dim * self.depth)) if flatten else X[idx]

    def port(self, class_name='RollingWindow'):
        """
        Port to C++
        :return: str
        """
        assert self.input_dim is not None, 'Unfitted'
        return jinja('preprocessing/RollingWindow.jinja', {
            'class_name': class_name,
            'depth': self.depth,
            'input_dim': self.input_dim
        })
