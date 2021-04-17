import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class MinMaxScaler(BaseStep):
    """
    Implementation of sklearn.ml.MinMaxScaler
    """
    def __init__(self, name='MinMaxScaler', num_features=-1):
        """
        :param name:
        :param num_features: int (0 for global, -1 for each feature, n for a specific number)
        """
        assert isinstance(num_features, int), 'ax MUST be an integer'

        super().__init__(name)
        self.num_features = num_features
        self.min = None
        self.max = None
        self.repeat = 1
        self.inplace = True

    def get_config(self):
        """
        Get config options
        """
        return {'num_features': self.num_features}

    def fit(self, X, y):
        """
        Learn min/max
        """
        self.set_X(X)

        if self.num_features == -1:
            self.num_features = self.input_dim

        if self.num_features == 0:
            self.min = X.min()
            self.max = X.max()
        else:
            assert (self.input_dim % self.num_features) == 0, 'num_features MUST be a divisor of X.shape[1]'
            
            mins = [X[:, i::self.num_features].min() for i in range(self.num_features)]
            maxs = [X[:, i::self.num_features].max() for i in range(self.num_features)]

            self.repeat = self.input_dim // self.num_features
            self.min = np.asarray(mins * self.repeat)
            self.max = np.asarray(maxs * self.repeat)

        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Transform
        :return: ndarray
        """
        assert self.min is not None and self.max is not None, 'Unfitted'

        return (X - self.min) / (self.max - self.min), y

    def get_template_data(self):
        """

        """
        return {
            'num_features': self.num_features,
            'min': self.min[:self.num_features] if self.num_features > 0 else self.min,
            'inv_range': 1 / (self.max - self.min)[:self.num_features] if self.num_features > 0 else 1 / (self.max - self.min),
        }
