from micromlgen import port
from sklearn.base import clone
from copy import deepcopy
from xgboost import XGBClassifier
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier
from eloquentarduino.ml.classification.device import ClassifierResources


class XGBClassifier2(SklearnClassifier):
    def __init__(self, *args, **kwargs):
        self.constructor_args = args
        self.constructor_kwargs = kwargs
        self.xgb = XGBClassifier(*args, **kwargs)

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memodict={}):
        return self.clone()

    def __getattr__(self, item):
        """
        Proxy all calls to sklearn classifier
        """
        return getattr(self.xgb, item)

    @property
    def sklearn_base(self):
        return type(self.xgb)

    def get_params(self, *args, **kwargs):
        try:
            return self.xgb.get_params(self, *args, **kwargs)
        except IndexError:
            return {}

    def clone(self):
        return XGBClassifier2(*self.constructor_args, **self.constructor_kwargs)

    def fit(self, X, y):
        """
        Fit
        """
        self.xgb.fit(X, y)
        # keep track of X and y
        self.X = X
        self.y = y

        return self

    def reset(self):
        """
        Reset the classifier
        """
        pass

    def hyperparameters_grid(self, X=None):
        """
        Get default hyperparameters values from grid search
        :param X:
        """
        return {}

    def port(self, classname=None, classmap=None, **kwargs):
        """
        Port to plain C++
        :param classname: str name of the ported class
        :param classmap: dict classmap in the format {class_idx: class_name}
        """
        ported = port(self.xgb, classname=classname, classmap=classmap, **kwargs)

        # replace #pragma once with #ifndef
        ported = ported.replace('#pragma once', '''
                #ifndef __CLASSIFIER__%(id)d
                #define __CLASSIFIER__%(id)d
                ''' % {'id': id(self)})

        return ported + '\n#endif'