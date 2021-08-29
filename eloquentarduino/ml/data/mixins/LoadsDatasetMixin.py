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
    def read_csv(
            cls,
            filename,
            columns=None,
            data_columns=None,
            label_column='y',
            skiprows=0,
            slice=None,
            **kwargs):
        """
        Read CSV file
        :param filename: str
        :param columns: list|str @deprecated, use data_columns
        :param data_columns: list|str list of columns to use as features
        :param label_column: str|None column to use as label
        :param skiprows: int how many rows to skip (@deprecated, ignored)
        :param slice: tuple (start, end) custom slicing of rows from each file
        """
        df = pd.read_csv(filename, **kwargs).select_dtypes(include=['number'])

        # keep compatibility
        if columns is not None:
            data_columns = data_columns or columns

        if data_columns is None:
            # if X_columns is None, use all columns but y_column
            data_columns = df.columns
        elif isinstance(data_columns, str):
            # if X_columns is a string, split on ","
            data_columns = [column.strip() for column in data_columns.split(',')]

        # remove label column from data columns
        data_columns = [column for column in data_columns if column != label_column]
        X = df[data_columns].to_numpy()

        if slice is not None:
            start, end = slice
            X = X[start:end]

        if label_column is not None:
            y = df[label_column].to_numpy().flatten()
        else:
            # if no y column is provided, fill with empty
            y = -np.ones(len(X))

        return cls(name=os.path.basename(filename), X=X, y=y, columns=data_columns)

    @classmethod
    def read_folder(
            cls,
            folder,
            data_columns=None,
            pattern=None,
            recursive=False,
            dataset_name='Dataset',
            skiprows=0,
            slice=None,
            **kwargs):
        """
        Read all files in a folder
        :param folder: str
        :param data_columns: list|str list of columns to use as features
        :param pattern: str regex to test files to be included
        :param recursive: bool if True, loads folder recursively
        :param dataset_name: str name of the dataset
        :param skiprows: int how many rows to skip (default=0)
        :param slice: tuple (start, end) custom slicing of rows from each file
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
            dataset = cls.read_csv(filename, data_columns=data_columns, label_column=None, skiprows=skiprows, slice=slice, **kwargs)
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
