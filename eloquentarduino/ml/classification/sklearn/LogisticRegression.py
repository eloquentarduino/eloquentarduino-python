from sklearn.linear_model import LogisticRegression as SklearnImplementation
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier


class LogisticRegression(SklearnClassifier, SklearnImplementation):
    """
    sklearn.linear_model.LogisticRegression wrapper
    """
    def hyperparameters_grid(self, X=None):
        """

        """
        return {
            'C': [0.01, 0.1, 1],
            'max_iter': [1e3, 1e4, 1e5]
        }