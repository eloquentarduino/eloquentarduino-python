from sklearn.linear_model import LogisticRegression as SklearnImplementation
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier


class LogisticRegression(SklearnClassifier, SklearnImplementation):
    """
    sklearn.linear_model.LogisticRegression wrapper
    """
    pass