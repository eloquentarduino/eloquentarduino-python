import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class MinMaxScaler(BaseStep):
    """
    Implementation of sklearn.ml.MinMaxScaler
    """
    def __init__(self, name='MinMaxScaler', ax=0):
        """
        :param name:
        :param ax: int {0: global; 1: for each feature; N: for each feature, flattened}
        """
        assert isinstance(ax, int), 'ax MUST be an integer'

        super().__init__(name)
        self.ax = ax
        self.min = None
        self.max = None
        self.repeat = 1
        self.inplace = True

    def fit(self, X, y):
        """
        Learn min/max
        """
        if self.ax == 0:
            self.min = X.min()
            self.max = X.max()
        elif self.ax == 1:
            self.min = X.min(axis=0)
            self.max = X.max(axis=0)
        else:
            mins = [X[:, i::self.ax].min() for i in range(self.ax)]
            maxs = [X[:, i::self.ax].max() for i in range(self.ax)]

            self.repeat = X.shape[1] // self.ax
            self.min = np.asarray(mins * self.repeat)
            self.max = np.asarray(maxs * self.repeat)

        self.set_X(X)

        return self.transform(X), y

    def transform(self, X):
        """
        Transform X
        :return: ndarray
        """
        assert self.min is not None and self.max is not None, 'Unfitted'
        return (X - self.min) / (self.max - self.min)

    def get_template_data(self):
        """

        """
        return {
            'ax': self.ax,
            'min': self.min,
            'inv_range': 1 / (self.max - self.min),
            'num_features': self.input_dim // self.repeat,
        }
