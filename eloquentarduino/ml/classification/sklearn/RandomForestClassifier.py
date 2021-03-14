from sklearn.ensemble import RandomForestClassifier as SklearnImplementation
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier


class RandomForestClassifier(SklearnClassifier, SklearnImplementation):
    """
    sklearn.ensemble.RandomForestClassifier wrapper
    """
    pass