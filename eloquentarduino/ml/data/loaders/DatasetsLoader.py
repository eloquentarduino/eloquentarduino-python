import re
import os.path
import numpy as np
from glob import glob
from sklearn.datasets import *
from eloquentarduino.ml.data import Dataset


class DatasetsLoader:
    """
    Load datasets from a variety of sources
    """
    def __init__(self):
        self.datasets = []

    def load_toy_datasets(self):
        """
        Load toy datasets from sklearn
        """
        self.datasets += [
            Dataset('Iris', *load_iris(return_X_y=True)),
            Dataset('Wine', *load_wine(return_X_y=True)),
            Dataset('Breast cancer', *load_breast_cancer(return_X_y=True)),
            Dataset('Digits', *load_digits(return_X_y=True)),
        ]

    def load_files(self, folder, pattern=r'\.csv$'):
        """
        Load datasets from files inside a folder
        :param folder: str
        :param pattern: regex pattern to match file names
        :param flags: str flags for regex
        """
        pattern = re.compile(pattern)

        for filename in glob('%s/*' % folder):
            if pattern.search(os.path.basename(filename)) is None:
                continue

            for delimiter in [',', '\t', ';', ' ']:
                try:
                    data = np.loadtxt(filename, delimiter=delimiter, skiprows=1)
                    X = data[:, :-1]
                    y = data[:, -1]
                    name = os.path.splitext(os.path.basename(filename))[0]
                    self.datasets.append(Dataset(name, X, y))
                    break
                except ValueError:
                    pass