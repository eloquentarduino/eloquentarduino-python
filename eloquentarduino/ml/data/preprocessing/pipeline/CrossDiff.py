import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class CrossDiff(BaseStep):
    """
    Compute difference among inputs
    """
    def __init__(self, name='CrossDiff'):
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
        for i in range(self.input_dim - 1):
            for j in range(i + 1, self.input_dim):
                X = np.hstack((X, (X[:, j] - X[:, i]).reshape((-1, 1))))

        return X, y

    def get_template_data(self):
        """

        """
        return {}