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
        return self.transform(X), y

    def transform(self, X):
        """
        Compute diff()
        :return: ndarray
        """
        return X[1:, :] - X[0:-1]

    def get_template_data(self):
        """

        """
        return {}