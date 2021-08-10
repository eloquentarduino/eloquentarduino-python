from math import ceil, sqrt
import numpy as np
from xgboost import XGBClassifier as XGBImplementation
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier


class XGBClassifier(SklearnClassifier, XGBImplementation):
    """
    xgboost.XGBClassifier wrapper
    """
    def __init__(self,
                 max_depth=None,
                 learning_rate=None,
                 n_estimators=100,
                 objective=None,
                 gamma=None,
                 min_child_weight=None,
                 max_delta_step=None,
                 subsample=None,
                 colsample_bytree=None,
                 colsample_bylevel=None,
                 colsample_bynode=None,
                 reg_alpha=None,
                 reg_lambda=None,
                 scale_pos_weight=None,
                 base_score=None,
                 random_state=None,
                 missing=np.nan,
                 num_parallel_tree=None,
                 monotone_constraints=None,
                 interaction_constraints=None,
                 importance_type="gain",
                 gpu_id=None,
                 validate_parameters=None,
                 **kwargs):
        """
        Patch constructor
        """
        super(XGBClassifier, self).__init__(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective=objective,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score,
            random_state=random_state,
            missing=missing,
            num_parallel_tree=num_parallel_tree,
            monotone_constraints=monotone_constraints,
            interaction_constraints=interaction_constraints,
            importance_type=importance_type,
            gpu_id=gpu_id,
            validate_parameters=validate_parameters,
            **kwargs)

    @property
    def sklearn_base(self):
        """
        Get xgboost implementation
        """
        return [base for base in self.__class__.__bases__ if base.__module__.startswith('xgboost.')][0]

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