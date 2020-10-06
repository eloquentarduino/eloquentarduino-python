from eloquentarduino.ml.data.preprocessing.BaseStep import BaseStep
from sklearn.preprocessing import PolynomialFeatures


class PolynomialFeaturesStep(BaseStep):
    """Apply 2° order polynomial features expansion"""
    def __init__(self, X, y, interaction_only=False):
        BaseStep.__init__(self, X, y)
        self.interaction_only = interaction_only

    @property
    def name(self):
        return 'polynomial_features'

    def __str__(self):
        return self.describe(('interaction_only', self.interaction_only))

    def transform(self):
        return self.apply(PolynomialFeatures(2, interaction_only=self.interaction_only).fit_transform(self.X)[:, 1:])

    def port(self):
        return self.jinja('PolynomialFeatures.jinja', {
            'interaction_only': self.interaction_only
        })