import numpy as np
from sklearn.covariance import EllipticEnvelope as SklearnImplementation
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class EllipticEnvelope(BaseStep):
    """
    Outlier detection using sklearn.EllipticEnvelop
    """
    def __init__(self, add_feature=False, predict=True, scale=1, name='EllipticEnvelope', random_state=0, **kwargs):
        """
        :param add_feature: bool (default=False) if True, a new column is added to the inputs with 0 if inlier, 1 if outlier
        :param predict: bool (default=True) if True, returns 0 for inliers, 1 for outliers
        """
        super().__init__(name=name)
        self.add_feature = add_feature
        self.predict = predict
        self.scale = scale
        self.envelope = SklearnImplementation(random_state=random_state, **kwargs)

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

        if self.predict:
            return X, is_outlier

        return X[~is_outlier], y[~is_outlier] if y is not None else None

    def get_template_data(self):
        """

        """
        VI = self.envelope.get_precision() * self.scale

        if np.abs(VI).min() > 1e3:
            VI = VI.astype(np.int)

        return {
            'add_feature': self.add_feature,
            'predict': self.predict,
            'v': self.envelope.location_,
            'VI': VI,
            'approx': VI.dtype == np.int,
            'offset': -self.envelope.offset_ * self.scale
        }
