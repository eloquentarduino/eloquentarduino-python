from sklearn.linear_model import PassiveAggressiveClassifier as SklearnImplementation
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier


class PassiveAggressiveClassifier(SklearnClassifier, SklearnImplementation):
    """
    sklearn.linear_model.PassiveAggressiveClassifier wrapper
    """
    pass