import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class RateLimit(BaseStep):
    """
    Skip inputs based on given frequency
    It only works on the C++ side
    """
    def __init__(self, skip, name='RateLimit'):
        """
        :param skip: int how many samples to skip
        """
        assert isinstance(skip, int) and skip > 0, 'skip MUST be a positive integer'
        super().__init__(name)
        self.skip = skip

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
        return X[::(self.skip + 1)], y[::(self.skip + 1)]

    def get_template_data(self):
        """

        """
        return {
            'skip': self.skip
        }