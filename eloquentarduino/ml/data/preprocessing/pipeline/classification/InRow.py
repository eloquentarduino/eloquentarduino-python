import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class InRow(BaseStep):
    """
    Only output a prediction when N predictions in a row agree
    """
    def __init__(self, n, name='InRow'):
        """
        :param n: int number of predictions to agree
        """
        super().__init__(name)
        self.n = n

    def fit(self, X, y):
        """
        Fit
        """
        assert X.shape[1] == 1, 'X MUST have a single column (it is interpreted as a classifier output)'

        self.set_X(X)

        # nothing to fit
        return self.transform(X, y)

    def transform(self, X, y=None):
        """

        """
        count = 0
        current = -1
        Xt = []
        yt = []

        for xi, yi in zip(X, y or []):
            if xi[0] != current:
                current = xi[0]
                count = 0

            count += 1

            if count == self.n:
                Xt.append(xi)
                yt.append(yi)

        return np.asarray(Xt), np.asarray(yt) if y is not None else None
