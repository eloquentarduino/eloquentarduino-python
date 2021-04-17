import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class Norm(BaseStep):
    """
    Implementation of sklearn.preprocessing.normalize
    """
    def __init__(self, name='Norm', norm='l2'):
        """
        Constructor
        :param name: str
        :param norm: str one of {l1, l2}
        """
        assert norm in ['l1', 'l2', 'inf'], 'norm MUST be one of {l1, l2, inf}'

        super().__init__(name)
        self.norm = norm

    def get_config(self):
        """
        Get config options
        """
        return {'norm': self.norm}

    def fit(self, X, y):
        """
        Fit
        """
        self.set_X(X)
        # nothing to fit
        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Transform
        """
        if self.norm == 'l2':
            return X / np.linalg.norm(X, axis=1).reshape((-1, 1)), y
        elif self.norm == 'l1':
            return X / np.sum(np.abs(X), axis=1).reshape((-1, 1)), y
        elif self.norm == 'inf':
            return X / np.max(np.abs(X), axis=1).reshape((-1, 1)), y

    def get_template_data(self):
        """
        Template data
        """
        return {
            'norm': self.norm
        }
