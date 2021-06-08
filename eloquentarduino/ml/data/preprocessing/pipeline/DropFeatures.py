import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class DropFeatures(BaseStep):
    """
    Drop features
    """
    def __init__(self, drop, name='DropFeatures'):
        """
        :param drop: list list of features to drop
        """
        assert isinstance(drop, list) and len(drop) > 0, 'drop MUST be a non-empty list'

        super().__init__(name)
        self.drop = drop
        self.keep = None

    def fit(self, X, y):
        """
        Fit
        """
        self.set_X(X)

        self.keep = [i for i in range(self.input_dim) if i not in self.drop]

        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        :return: ndarray
        """
        assert self.keep is not None, 'Unfitted'

        return X[:, self.keep], y

    def get_template_data(self):
        """

        """
        return {
            'keep': self.keep
        }