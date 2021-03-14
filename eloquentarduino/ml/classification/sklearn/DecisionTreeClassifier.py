from sklearn.tree import DecisionTreeClassifier as SklearnImplementation
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier


class DecisionTreeClassifier(SklearnClassifier, SklearnImplementation):
    """
    sklearn.tree.DecisionTree wrapper
    """
    pass