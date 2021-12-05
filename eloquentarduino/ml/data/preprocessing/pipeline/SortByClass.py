import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class SortByClass(BaseStep):
    """
    Test if sorting dataset by class influences results
    Python only!
    @added 0.1.19
    """
    def __init__(self, name='SortByClass'):
        """
        Constructor
        """
        super().__init__(name)

    def fit(self, X, y):
        """
        Fit
        """
        self.set_X(X)

        indices = np.argsort(y)

        return X[indices], y[indices] if y is not None else None

    def transform(self, X, y=None):
        """
        Transform
        Do nothing
        This is a fit-only transformer!
        """
        return X, y


    def get_template_data(self):
        """
        Template data
        """
        return {
        }