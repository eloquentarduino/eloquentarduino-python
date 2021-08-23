import re
import os.path
import numpy as np
import pandas as pd
from glob import glob
from sklearn.datasets import *


class LoadsDatasetMixin:
    """
    Mixin to load datasets from files and folders using pandas' API
    Made to be used by Dataset
    """
    @classmethod
    def read_csv(cls, filename, X_columns=None, y_column='y', **kwargs):
        """
        Read CSV file
        :param filename: str
        :param X_columns: list|str list of columns to use as features
        :param y_column: str|None column to use as label
        """
        df = pd.read_csv(filename, **kwargs).select_dtypes(include=['number'])

        if X_columns is None:
            # if X_columns is None, use all columns but y_column
            X_columns = [column for column in df.columns if column != y_column]
        elif isinstance(X_columns, str):
            # if X_columns is a string, split on ","
            X_columns = [column.strip() for column in X_columns.split(',')]

        X = df[X_columns].to_numpy()

        if y_column is not None:
            y = df[y_column].to_numpy().flatten()
        else:
            # if no y column is provided, fill with empty
            y = -np.ones(len(X))

        return cls(name=os.path.basename(filename), X=X, y=y, columns=X_columns)

    @classmethod
    def read_folder(cls, folder, X_columns=None, pattern=None, recursive=False, dataset_name='Dataset', **kwargs):
        """
        Read all files in a folder
        :param folder: str
        :param X_columns: list|str list of columns to use as features
        :param pattern: str regex to test files to be included
        :param recursive: bool if True, loads folder recursively
        :param dataset_name: str name of the dataset
        """
        if not folder.endswith('/'):
            folder += '/'

        if os.path.isdir(folder):
            folder += '*'

        Xs = []
        ys = []
        columns = None
        classmap = {}
        filenames = glob(folder, recursive=recursive)

        if pattern is not None:
            # filter files by regex
            filenames = [filename for filename in filenames if re.search(pattern, os.path.basename(filename)) is not None]

        assert len(filenames) > 0, 'no file found'

        for i, filename in enumerate(sorted(filenames)):
            dataset = cls.read_csv(filename, X_columns=X_columns, y_column=None, **kwargs)
            Xs.append(dataset.X)
            ys.append(np.ones(len(dataset.y)) * i)
            classmap[i] = os.path.splitext(os.path.basename(filename))[0]
            columns = dataset.columns

        X = np.vstack(Xs)
        y = np.concatenate(ys)

        return cls(name=dataset_name, X=X, y=y, classmap=classmap, columns=columns)

    @classmethod
    def Iris(cls):
        """
        Create the Iris dataset
        """
        return cls('Iris', *load_iris(return_X_y=True))

    @staticmethod
    def MNIST_Tensorflow(cls):
        """
        Create the MNIST dataset formatted for Tensorflow
        """
        X, y = load_digits(return_X_y=True)

        return cls('MNIST Tf', np.expand_dims(X.reshape((-1, 8, 8)), -1), y, test_validity=False)
