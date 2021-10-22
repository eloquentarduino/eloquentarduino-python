import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class InRow(BaseStep):
    """
    Only output a prediction when N predictions in a row agree
    """
    def __init__(self, n, unsure_class=None, name='InRow'):
        """
        :param n: int number of predictions to agree
        :param unsure_class: int|None if int, this step will also output "uncertain" class
        """
        assert n <= 255, 'n MUST be <= 255'

        super().__init__(name)
        self.n = n
        self.unsure_class = unsure_class
        self.missing_rate = 0

    def fit(self, X, y):
        """
        Fit
        """
        assert X.shape[1] == 1, 'X MUST have a single column (it is interpreted as a classifier output)'

        self.set_X(X)

        # nothing to fit
        return self.transform(X, y)

    def transform(self, X, y=None, holes=False):
        """
        :param X: np.ndarray
        :param y: np.ndarray
        :param holes: if True, appends np.nan to result when not in row
        """
        count = 0
        current = -1
        predictions = 0
        Xt = []
        yt = []

        for xi, yi in zip(X, y if y is not None else []):
            if xi[0] != current:
                current = xi[0]
                count = 0

            count += 1

            if count >= self.n:
                predictions += 1

                Xt.append(xi)
                yt.append(yi)
            elif self.unsure_class:
                Xt.append([self.unsure_class])
                yt.append(yi)
            elif holes:
                Xt.append([np.nan])
                yt.append(yi)

        # keep track of missing rate
        self.missing_rate = 1 - predictions / max(1, len(X))

        return np.asarray(Xt), np.asarray(yt) if y is not None else None

    def get_config(self):
        """

        """
        return {
            'n': self.n
        }

    def get_template_data(self):
        """

        """
        return {
            'n': self.n
        }
