import numpy as np
from eloquentarduino.utils import jinja


class RollingWindow:
    """
    Process data as a rolling window
    """
    def __init__(self, depth):
        """
        :param depth: how many samples will form a single window
        """
        assert depth > 1, "depth MUST be greater than 1"
        self.depth = depth
        self.input_dim = None

    def fit(self, X):
        """
        "Fit" the rolling window
        """
        self.transform(X)

    def transform(self, X):
        """
        Transform data
        :param X: input data
        :return: np.ndarray transformed input
        """
        self.input_dim = X.shape[1]
        w = np.arange(self.depth)
        t = np.arange(len(X) - self.depth + 1)
        idx = (w + t.reshape((-1, 1)))
        return X[idx].reshape((-1, self.input_dim * self.depth))

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
