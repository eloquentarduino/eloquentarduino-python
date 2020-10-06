import numpy as np
from eloquentarduino.ml.data.preprocessing.BaseStep import BaseStep
from sklearn.feature_selection import SelectKBest, chi2


class SelectKBestStep(BaseStep):
    """Feature selection with scikit-learn's SelectKBest"""
    def __init__(self, X, y, k, score_function=None):
        BaseStep.__init__(self, X, y)
        assert k > 0, "k MUST be > 0"
        self.k = k
        self.score_function = score_function or chi2
        self.kbest = SelectKBest(self.score_function, k=self.k)

    @property
    def name(self):
        return 'select_kbest'

    def __str__(self):
        return self.describe(('k', self.k), ('score_function', self.score_function))

    def transform(self):
        return self.apply(self.kbest.fit_transform(self.X, self.y))

    def port(self):
        idx = (-self.kbest.scores_).argsort()[:self.k]
        idx = np.sort(idx)
        return self.jinja('SelectKBest.jinja', {
            'idx': idx
        })