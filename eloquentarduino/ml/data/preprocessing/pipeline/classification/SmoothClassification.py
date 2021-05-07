import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class SmoothClassification(BaseStep):
    """
    Apply classification
    """
    def __init__(self, decay=0.8, mean_thresh=0.05, var_thresh=0.05, name='SmoothClassification'):
        """
        """
        super().__init__(name)
        self.k = 0
        self.mean = 0
        self.var = 0
        self.decay = decay
        self.mean_thresh = mean_thresh
        self.var_thresh = var_thresh

    def fit(self, X, y):
        """
        Fit
        """
        assert X.shape[1] == 1, 'X MUST have a single column (it is interpreted as a classifier output)'

        self.set_X(X)

        # nothing to fit
        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Transform
        """
        self.k = 0
        self.mean = 0
        self.var = 0

        x_smooth = []

        for x in X[:, 0]:
            self._push(x)

            if self.k > 1 and abs(x - self.mean) < self.mean_thresh and self.var < self.var_thresh:
                x_smooth.append(x)

        x_smooth = np.asarray(x_smooth, dtype=np.float32)

        return x_smooth.reshape((-1, 1)), x_smooth

    def get_template_data(self):
        """

        """
        return {
            'decay': self.decay,
            'mean_thresh': self.mean_thresh,
            'var_thresh': self.var_thresh
        }

    def _push(self, x):
        """
        Push new element
        """
        self.k += 1
        mean = self.decay * self.mean + (1 - self.decay) * x

        if self.k > 1:
            # running variance
            var = self.var + (x - self.mean) * (x - mean)
            self.var = var / self.k

        self.mean = mean
