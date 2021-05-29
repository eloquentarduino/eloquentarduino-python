import numpy as np
from sklearn.covariance import EllipticEnvelope as SklearnImplementation
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class EllipticEnvelope(BaseStep):
    """
    Outlier detection using sklearn.EllipticEnvelop
    """
    def __init__(self, add_feature=False, name='EllipticEnvelope', **kwargs):
        """
        :param add_feature: bool (default=False) if True, a new column is added to the inputs with 0 if inlier, 1 if outlier
        """
        super().__init__(name=name)
        self.add_feature = add_feature
        self.envelope = SklearnImplementation(**kwargs)

    def fit(self, X, y):
        """

        """
        self.set_X(X)
        self.envelope.fit(X)

        return self.transform(X, y)

    def transform(self, X, y=None):
        """

        """
        is_outlier = self.envelope.predict(X) < 0

        if self.add_feature:
            return np.hstack((X, is_outlier.reshape((-1, 1)))), y

        return X[~is_outlier], y[~is_outlier] if y is not None else None

    def get_template_data(self):
        """

        """
        return {
            'v': self.envelope.location_,
            'VI': self.envelope.get_precision(),
            'offset': -self.envelope.offset_
        }
