from sklearn.naive_bayes import GaussianNB as SklearnImplementation
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier


class GaussianNB(SklearnClassifier, SklearnImplementation):
    """
    sklearn.naive_bayes.GaussianNB wrapper
    """
    def hyperparameters_grid(self, X=None):
        return {
            'var_smoothing': [1e-5, 1e-7, 1e-9]
        }