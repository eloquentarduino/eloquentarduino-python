from sklearn.linear_model import PassiveAggressiveClassifier as SklearnImplementation
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier


class PassiveAggressiveClassifier(SklearnClassifier, SklearnImplementation):
    """
    sklearn.linear_model.PassiveAggressiveClassifier wrapper
    @todo port to C++
    """
    def hyperparameters_grid(self, X=None):
        return {
            'C': [0.01, 0.1, 1],
            'shuffle': [True, False]
        }