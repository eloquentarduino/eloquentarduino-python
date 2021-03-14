from sklearn.naive_bayes import GaussianNB as SklearnImplementation
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier


class GaussianNB(SklearnClassifier, SklearnImplementation):
    """
    sklearn.naive_bayes.GaussianNB wrapper
    """
    pass