import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class Diff(BaseStep):
    """
    Compute diff() of dataset
    """
    def __init__(self, name='Diff'):
        super().__init__(name)
        self.inplace = True

    def fit(self, X, y):
        """
        Fit
        """
        self.set_X(X)
        # nothing to fit
        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Compute diff()
        :return: ndarray
        """
        return np.vstack((X[0].reshape((1, -1)), X[1:, :] - X[0:-1])), y

    def get_template_data(self):
        """

        """
        return {}