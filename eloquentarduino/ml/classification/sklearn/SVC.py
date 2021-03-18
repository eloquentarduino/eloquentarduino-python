from sklearn.svm import SVC as SklearnImplementation
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier


class SVC(SklearnClassifier, SklearnImplementation):
    """
    sklearn.tree.DecisionTree wrapper
    """
    def hyperparameters_grid(self, X=None):
        """

        """
        return {
            'kernel': ['poly', 'rbf'],
            'degree': [2, 3],
            'gamma': [0.001, 0.1, 1, 'auto'],
            'C': [0.01, 0.1, 1],
            'max_iter': [1000, 10000]
        }