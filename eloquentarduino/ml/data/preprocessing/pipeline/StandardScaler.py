import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class StandardScaler(BaseStep):
    """
    Implementation of sklearn.ml.StandardScaler
    """
    def __init__(self, name='StandardScaler', ax=0):
        """
        :param name:
        :param ax: int {0: global; 1: for each feature; N: for each feature, flattened}
        """
        assert isinstance(ax, int), 'ax MUST be an integer'

        super().__init__(name)
        self.ax = ax
        self.mean = None
        self.std = None
        self.repeat = 1
        self.inplace = True

    def fit(self, X, y):
        """
        Learn mean/std
        """
        if self.ax == 0:
            self.mean = X.mean()
            self.std = X.std()
        elif self.ax == 1:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
        else:
            mean = [X[:, i::self.ax].min() for i in range(self.ax)]
            std = [X[:, i::self.ax].max() for i in range(self.ax)]

            self.repeat = X.shape[1] // self.ax
            self.mean = np.asarray(mean * self.repeat)
            self.std = np.asarray(std * self.repeat)

        self.set_X(X)

        return self.transform(X), y

    def transform(self, X):
        """
        Transform X
        :return: ndarray
        """
        assert self.mean is not None and self.std is not None, 'Unfitted'
        return (X - self.mean) / self.std

    def get_template_data(self):
        """

        """
        return {
            'ax': self.ax,
            'mean': self.mean,
            'inv_std': 1 / self.std,
            'num_features': self.input_dim // self.repeat,
        }
