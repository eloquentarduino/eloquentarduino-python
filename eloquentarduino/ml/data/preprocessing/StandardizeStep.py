import numpy as np
from eloquentarduino.ml.data.preprocessing.BaseStep import BaseStep


class StandardizeStep(BaseStep):
    """Apply standardization"""
    def __init__(self, X, y, featurewise=False):
        BaseStep.__init__(self, X, y)
        self.featurewise = int(featurewise)
        self.mean = None
        self.std = None

    @property
    def name(self):
        return 'standardize'

    def __str__(self):
        return self.describe(('featurewise', self.featurewise))

    def transform(self):
        if not self.featurewise:
            self.mean = self.X.mean()
            self.std = self.X.std()
        elif self.featurewise == 1:
            self.mean = self.X.mean(axis=0)
            self.std = self.X.std(axis=0)
        else:
            means = [self.X[:, i::self.featurewise].mean() for i in range(self.featurewise)]
            stds = [self.X[:, i::self.featurewise].std() for i in range(self.featurewise)]
            self.mean = np.asarray(means * (self.input_dim // self.featurewise))
            self.std = np.asarray(stds * (self.input_dim // self.featurewise))
        return self.apply((self.X - self.mean) / self.std)

    def port(self):
        return self.jinja('Standardize.jinja', {
            'xmean': self.mean,
            'inverse_std': 1 / self.std,
            'featurewise': self.featurewise
        })
