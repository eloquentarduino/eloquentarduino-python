import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class StandardScaler(BaseStep):
    """
    Implementation of sklearn.ml.StandardScaler
    """
    def __init__(self, name='StandardScaler', num_features=-1):
        """
        :param name:
        :param num_features: int {0: global; 1: for each feature; N: for each feature, flattened}
        """
        assert isinstance(num_features, int), 'ax MUST be an integer'

        super().__init__(name)
        self.num_features = num_features
        self.mean = None
        self.std = None
        self.repeat = 1
        self.inplace = True

    def get_config(self):
        """
        Get config options
        """
        return {'num_features': self.num_features}

    def fit(self, X, y):
        """
        Learn mean/std
        """
        self.set_X(X)

        if self.num_features == -1:
            self.num_features = self.input_dim

        if self.num_features == 0:
            self.mean = X.mean()
            self.std = X.std()
        else:
            assert self.input_dim % self.num_features == 0, 'num_features MUST be a divisor of X.shape[1]'

            mean = [X[:, i::self.num_features].mean() for i in range(self.num_features)]
            std = [X[:, i::self.num_features].std() for i in range(self.num_features)]

            self.repeat = self.input_dim // self.num_features
            self.mean = np.asarray(mean * self.repeat)
            self.std = np.asarray(std * self.repeat)

        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Transform
        """
        assert self.mean is not None and self.std is not None, 'Unfitted'

        return (X - self.mean) / self.std, y

    def get_template_data(self):
        """

        """
        return {
            'mean': self.mean[:self.num_features],
            'inv_std': 1 / self.std[:self.num_features],
            'num_features': self.num_features,
        }
