from math import ceil, sqrt
from xgboost import XGBClassifier as SklearnImplementation
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier


class XGBClassifier(SklearnClassifier, SklearnImplementation):
    """
    xgboost.XGBClassifier wrapper
    """
    def hyperparameters_grid(self, X=None):
        if X is None:
            return {
                'n_estimators': [10, 25, 50],
                'max_depth': [10, 30, None],
                'min_samples_leaf': [5, 10, 20],
                'max_features': [0.5, 0.75, "sqrt", None],
                'gamma': [0, 1, 10],
            }

        num_samples, num_features = X.shape[:2]

        return {
            'n_estimators': [10, 25, 50],
            'max_depth': set([max(2, ceil(num_features / 5)), ceil(sqrt(num_features)), None]),
            'min_samples_leaf': set([5, ceil(num_samples / 100), ceil(num_samples / 30)]),
            'max_features': [0.5, 0.75, "sqrt", None],
            'gamma': [0, 1, 10],
            'eta': [0.1, 0.3, 0.7]
        }