from math import sqrt, ceil
from sklearn.tree import DecisionTreeClassifier as SklearnImplementation
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier


class DecisionTreeClassifier(SklearnClassifier, SklearnImplementation):
    """
    sklearn.tree.DecisionTree wrapper
    """
    def hyperparameters_grid(self, X=None):
        if X is None:
            return {
                'max_depth': [10, 30, 50],
                'min_samples_leaf': [5, 10, 20],
                'max_features': [0.5, "sqrt", None]
            }

        num_samples, num_features = X.shape[:2]

        return {
            'max_depth': set([max(2, ceil(num_features / 5)), ceil(sqrt(num_features)), num_features * 2]),
            'min_samples_leaf': set([5, ceil(num_samples / 100), ceil(num_samples / 30)]),
            'max_features': [0.5, "sqrt", None]
        }