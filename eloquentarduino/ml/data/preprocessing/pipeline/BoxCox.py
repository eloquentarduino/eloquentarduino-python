from sklearn.preprocessing import PowerTransformer
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class BoxCox(BaseStep):
    """
    sklearn.PowerTransform(method='box-cox') implementation
    """
    def __init__(self, name='BoxCox'):
        super().__init__(name)
        self.inplace = True
        self.power = PowerTransformer(method='box-cox', standardize=False)

    def fit(self, X, y):
        """
        Fit
        """
        self.set_X(X)
        self.power.fit(X)

        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Transform
        """
        return self.power.transform(X), y

    def get_template_data(self):
        """
        Get template data
        """
        return {
            'lambdas': self.power.lambdas_,
            'has_zeros': len([l for l in self.power.lambdas_ if l == 0])
        }