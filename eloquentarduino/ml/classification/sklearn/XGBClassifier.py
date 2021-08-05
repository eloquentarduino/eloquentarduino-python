from math import ceil, sqrt
from copy import deepcopy
from xgboost import XGBClassifier as XGBImplementation
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier


class XGBClassifier(SklearnClassifier, XGBImplementation):
    """
    xgboost.XGBClassifier wrapper
    """
    def __init__(self, random_state=0, objective='multi:softprob', missing=None, **kwargs):
        """
        Patch constructor
        """
        super().__init__(random_state=random_state, objective=objective, missing=missing, **kwargs)

    @property
    def sklearn_base(self):
        """
        Get xgboost implementation
        """
        return [base for base in self.__class__.__bases__ if base.__module__.startswith('xgboost.')][0]

    def clone(self):
        return XGBClassifier(**self.get_params())

    def fit(self, X, y):
        """
        Fit
        """
        self.sklearn_base.set_params(self, num_class=len(set(y)))
        self.sklearn_base.fit(self, X, y)
        # keep track of X and y
        self.X = X
        self.y = y

        return self

    def hyperparameters_grid(self, X=None):
        """

        """
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