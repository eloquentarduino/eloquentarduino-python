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
        Xt = []
        yt = []

        for xi, yi in zip(X, y if y is not None else []):
            if xi[0] != current:
                current = xi[0]
                count = 0

            count += 1

            if count >= self.n:
                Xt.append(xi)
                yt.append(yi)
            elif self.unsure_class:
                Xt.append([self.unsure_class])
                yt.append(yi)
            elif holes:
                Xt.append([np.nan])
                yt.append(yi)

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
