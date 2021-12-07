from sefr import SEFR as NativeImplementation
from eloquentarduino.ml.classification.abstract.Classifier import Classifier


class SEFR(Classifier, NativeImplementation):
    """
    sefr.SEFR wrapper
    @todo port to C++
    """
    @property
    def base(self):
        return [base for base in self.__class__.__bases__ if base.__module__.startswith('sefr.')][0]

    def fit(self, X, y):
        """
        Fit
        """
        self.base.fit(self, X, y)

        # keep track of X and y
        self.X = X
        self.y = y

        return self

    def hyperparameters_grid(self, X=None):
        """
        No hyper-parameter
        :param X:
        :return: dict
        """
        return {

        }