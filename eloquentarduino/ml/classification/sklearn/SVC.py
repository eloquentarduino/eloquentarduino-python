from sklearn.svm import SVC as SklearnImplementation
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier


class SVC(SklearnClassifier, SklearnImplementation):
    """
    sklearn.tree.DecisionTree wrapper
    """
    pass