import numpy as np
from eloquentarduino.ml.data.preprocessing.BaseStep import BaseStep


class NormalizeStep(BaseStep):
    """Apply min/max normalization"""
    def __init__(self, X, y, featurewise=False):
        BaseStep.__init__(self, X, y)
        self.featurewise = int(featurewise)
        self.min = None
        self.max = None

    @property
    def name(self):
        return 'normalize'

    def __str__(self):
        return self.describe(('featurewise', self.featurewise))

    def transform(self):
        if not self.featurewise:
            self.min = self.X.min()
            self.max = self.X.max()
        elif self.featurewise == 1:
            self.min = self.X.min(axis=0)
            self.max = self.X.max(axis=0)
        else:
            mins = [self.X[:, i::self.featurewise].min() for i in range(self.featurewise)]
            maxs = [self.X[:, i::self.featurewise].max() for i in range(self.featurewise)]
            self.min = np.asarray(mins * (self.input_dim // self.featurewise))
            self.max = np.asarray(maxs * (self.input_dim // self.featurewise))
        return self.apply((self.X - self.min) / (self.max - self.min))

    def port(self):
        return self.jinja('Normalize.jinja', {
            'xmin': self.min,
            'inverse_range': 1 / (self.max - self.min),
            'featurewise': self.featurewise
        })