import numpy as np
from eloquentarduino.ml.data.preprocessing.BaseStep import BaseStep


class UnitStep(BaseStep):
    """Apply unitary length normalization"""
    def __init__(self, X, y):
        BaseStep.__init__(self, X, y)

    @property
    def name(self):
        return 'unit'

    def __str__(self):
        return self.describe()

    def transform(self):
        return self.apply((self.X / np.linalg.norm(self.X, axis=1).reshape(-1, 1)))

    def port(self):
        return self.jinja('Unit.jinja')