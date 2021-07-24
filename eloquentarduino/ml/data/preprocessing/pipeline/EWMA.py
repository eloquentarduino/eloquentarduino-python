import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class EWMA(BaseStep):
    """
    Exponentially weightened moving average
    """
    def __init__(self, decay, name='EWMA'):
        """
        :param decay: float decay factor
        """
        assert 0 < decay <= 1, 'alpha MUST be in the range ]0, 1]'

        super().__init__(name)
        self.decay = decay

    def fit(self, X, y):
        """

        """
        self.set_X(X)

        return self.transform(X, y)

    def transform(self, X, y=None):
        """

        """
        prev = np.zeros(X.shape[1])
        ewma = np.empty_like(X)

        for i, xi in enumerate(X):
            curr = xi * self.decay + prev * (1 - self.decay)
            ewma[i] = prev = curr

        return ewma, y

    def get_template_data(self):
        """

        """
        return {
            'decay': self.decay
        }

    def get_config(self):
        """

        """
        return {
            'decay': self.decay
        }
