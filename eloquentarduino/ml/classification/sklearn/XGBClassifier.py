from xgboost import XGBClassifier as SklearnImplementation
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier


class XGBClassifier(SklearnClassifier, SklearnImplementation):
    """
    xgboost.XGBClassifier wrapper
    """
    pass