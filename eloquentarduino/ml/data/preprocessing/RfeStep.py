import numpy as np
from eloquentarduino.ml.data.preprocessing.BaseStep import BaseStep
from sklearn.feature_selection import RFE


class RfeStep(BaseStep):
    """Feature selection with scikit-learn's RFE"""
    def __init__(self, X, y, estimator, k):
        BaseStep.__init__(self, X, y)
        assert k > 0, "k MUST be > 0"
        self.k = k
        self.estimator = estimator
        self.rfe = RFE(self.estimator, n_features_to_select=self.k)

    @property
    def name(self):
        return 'rfe'

    def transform(self):
        return self.apply(self.rfe.fit_transform(self.X, self.y))

    def port(self):
        idx = self.rfe.ranking_.argsort()[:self.k]
        idx = np.sort(idx)
        return self.jinja('RFE.jinja', {
            'idx': idx
        })