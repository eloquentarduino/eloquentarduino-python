import numpy as np
from functools import reduce
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
        return reduce(lambda x, prod: x * prod, self.X.shape[1:], 1)

    @property
    def num_classes(self):
        """
        Get number of classes
        :return: int
        """
        # account for one-hot encoding
        return len(np.unique(self.y)) if len(self.y.shape) == 1 else self.y.shape[1]

    def shuffle(self, **kwargs):
        """
        Shuffle X and y
        :return: Dataset
        """
        self.X, self.y = shuffle(self.X, self.y, **kwargs)

        return self

    def random(self, size):
        """
        Get random samples
        :param size: int number of samples to return
        """
        idx = np.random.permutation(self.length)[:size]
        return self.X[idx], self.y[idx]

    def split(self, test=0, validation=0, return_empty=True, shuffle=True):
        """
        Split array into train, validation, test
        :param test: float test size percent
        :param validation: float validation size percent
        :param return_empty: bool if empty splits should be returned
        :param shuffle: bool if dataset should be shuffled before splitting
        """
        assert test > 0 or validation > 0, 'either test or validation MUST be greater than 0'
        assert test + validation < 1, 'test + validation MUST be less than 0'
        assert isinstance(return_empty, bool), 'return_empty MUST be a boolean'
        assert isinstance(shuffle, bool), 'shuffle MUST be a boolean'

        train_split = int(self.length * (1 - test - validation))
        validation_split = int(self.length * validation) + train_split

        if shuffle:
            self.shuffle()

        x_train, x_valid, x_test = np.split(self.X, [train_split, validation_split])
        y_train, y_valid, y_test = np.split(self.y, [train_split, validation_split])

        arrays = [x_train, y_train, x_valid, y_valid, x_test, y_test]

        if not return_empty:
            arrays = [arr for arr in arrays if len(arr) > 0]

        return arrays
