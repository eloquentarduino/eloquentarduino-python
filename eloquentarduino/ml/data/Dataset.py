import numpy as np
from sklearn.utils import shuffle


class Dataset:
    """
    Abstraction of a dataset
    """
    def __init__(self, name, X, y):
        """
        :param name:
        :param X:
        :param y:
        """
        assert len(X) == len(y), 'X and y length MUST match'
        self.name = name
        self.X = X
        self.y = y

    @property
    def length(self):
        """
        Get dataset length
        """
        return len(self.X)

    @property
    def num_features(self):
        """
        Get number of features of X
        :return: int
        """
        return self.X.shape[1]

    @property
    def num_classes(self):
        """
        Get number of classes
        :return: int
        """
        return len(np.unique(self.y))

    def shuffle(self, **kwargs):
        """
        Shuffle X and y
        :return: Dataset
        """
        self.X, self.y = shuffle(self.X, self.y, **kwargs)

        return self
