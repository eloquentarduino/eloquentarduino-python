import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep
from sklearn.feature_selection import RFE as SklearnRFE


class RFE(BaseStep):
    """
    Implementation of sklearn.feature_selection.RFE
    """
    def __init__(self, estimator, k, name='RFE'):
        """
        Constructor
        :param estimator: classifier
        :param k: int k best features
        """
        assert isinstance(k, int) and k > 0, 'k MUST be positive'

        super().__init__(name)
        self.k = k
        self.rfe = SklearnRFE(estimator=estimator, n_features_to_select=k)
        self.idx = None
        self.inplace = True

    def get_config(self):
        """
        Get config options
        """
        return {'k': self.k}

    def fit(self, X, y):
        """
        Fit
        """
        self.set_X(X)
        self.rfe.fit(X, y)
        self.idx = self.rfe.ranking_.argsort()[:self.k]
        self.idx = np.sort(self.idx)

        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Transform
        """
        return X[:, self.idx], y

    def get_template_data(self):
        """
        Template data
        """
        return {
            'k': self.k,
            'idx': self.idx
        }