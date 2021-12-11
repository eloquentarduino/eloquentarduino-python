import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep
from sklearn.decomposition import PCA as SklearnPCA


class PCA(BaseStep):
    """
    Implementation of sklearn.decomposition.PCA
    @todo port to C++
    """
    def __init__(self, k, name='PCA'):
        """
        Constructor
        :param k: int
        """
        assert isinstance(k, int) and k > 0, 'k MUST be positive'

        super().__init__(name)
        self.k = k
        self.pca = SklearnPCA(n_components=k)

    def get_config(self):
        """
        Get config options
        """
        return {
            'k': self.k
        }

    def fit(self, X, y):
        """
        Fit
        """
        self.set_X(X)
        self.pca.fit(X)

        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Transform
        """
        return self.pca.transform(X), y

    def get_template_data(self):
        """
        Template data
        """
        return {
            'k': self.k,
            # @todo coefs
        }