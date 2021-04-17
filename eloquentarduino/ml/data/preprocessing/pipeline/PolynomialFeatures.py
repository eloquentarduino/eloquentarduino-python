from sklearn.preprocessing import PolynomialFeatures as Poly
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class PolynomialFeatures(BaseStep):
    """
    Implementation of sklearn.preprocessing.PolynomialFeatures
    """
    def __init__(self, name='PolynomialFeatures', interaction_only=False):
        super().__init__(name)
        self.interaction_only = interaction_only

    def fit(self, X, y):
        """
        Fit
        """
        self.set_X(X)
        # nothing to fit
        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Transform
        """
        # skip initial 1
        return Poly(2, interaction_only=self.interaction_only).fit_transform(X)[:, 1:], y

    def get_template_data(self):
        """
        Get template data
        """
        return {
            'interaction_only': self.interaction_only
        }