import numpy as np
from scipy.stats import mode
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class Window(BaseStep):
    """
    Sliding window
    """
    def __init__(self, length, shift=1, flatten=True, name='Window'):
        """
        :param length: int
        :param shift: int|float complement to overlap
        :param flatten: bool
        """
        assert isinstance(length, int) and length > 1, 'length MUST be <= 1'
        assert isinstance(shift, int) and shift <= length or isinstance(shift, float) and 0 < shift <= 1, 'shift MUST be an integer <= length or a float in the range ]0, 1]'

        super().__init__(name)
        self.length = length
        self.shift = shift if isinstance(shift, int) else int(length * shift)
        self.flatten = flatten

    def get_config(self):
        """
        Get config options
        """
        return {'length': self.length, 'shift': self.shift}

    def fit(self, X, y):
        """
        Fit
        """
        self.set_X(X)
        self.working_dim = self.input_dim * self.length

        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Transform
        """
        idx = self.idx(X)

        return X[idx].reshape((-1, self.input_dim * self.length)) if self.flatten else X[idx], np.asarray([mode(window)[0][0] for window in y[idx]])

    def idx(self, X):
        """
        Get dense indices of X array
        :param X:
        :return: ndarray
        """
        w = np.arange(self.length)
        t = np.arange(len(X) - self.length + 1)
        idx = (w + t.reshape((-1, 1)))

        return idx[::self.shift]

    def get_template_data(self):
        """
        Get template data
        """
        return {
            'length': self.length,
            'shift': self.shift,
        }