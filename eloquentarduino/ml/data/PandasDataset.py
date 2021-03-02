import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.decomposition import PCA


Split = namedtuple('Split', 'name df np indices')


class PandasDataset:
    """
    Load dataset from a pandas dataframe
    """
    def __init__(self, df, columns=None):
        """
        Constructor
        :param df: pd.DataFrame|string dataframe or path to file
        :param columns: list list of columns
        """
        assert isinstance(df, pd.DataFrame) or isinstance(df, str), 'df MUST be a DataFrame or a string'
        assert columns is None or isinstance(columns, list) or isinstance(columns, tuple), 'columns MUST be None or a list'

        self.df = df if isinstance(df, pd.DataFrame) else pd.read_csv(df)
        self.columns = columns or self.df.columns
        self.df = self.df[columns]
        self.splits = []
        self.every = 1

    @property
    def length(self):
        """
        Length of DataFrame
        """
        return len(self.df)

    @property
    def X(self):
        """
        Get feature vectors
        :return: np.ndarray
        """
        return np.vstack([split.np for split in self.splits])

    @property
    def y(self):
        """
        Get labels
        :return: np.ndarray
        """
        return np.concatenate([np.ones(len(split.np)) * i for i, split in enumerate(self.splits)])

    @property
    def classmap(self):
        """
        Return dataset classmap
        :return: dict
        """
        return {i: split.name for i, split in enumerate(self.splits)}

    def describe(self, *args, **kwargs):
        """
        Describe DataFrame
        """
        return self.df.describe(*args, **kwargs)

    def diff(self):
        """
        Return a new PandasDataset with the diff() from the current DataFrame
        :return: PandasDataset
        """
        clone = PandasDataset(self.df.diff(), self.columns)

        for split in self.splits:
            clone.add_split(split.name, *split.indices)

        return clone

    def once_every(self, n):
        """
        Only keep one sample every n
        :param n: int
        :return: self
        """
        self.every = n
        self.df = self.df[::n]

        return self

    def add_split(self, name, *args):
        """
        Split dataset into chunks based on position
        :param name: str
        """
        df = None
        indices = []

        for start, end in args:
            start = start // self.every
            end = end // self.every
            chunk = self.df[start:end].reset_index(drop=True)
            df = chunk if df is None else df.append(chunk).reset_index(drop=True)
            indices.append((start, end))

        self.splits.append(Split(name=name, df=df, np=df.dropna(axis=1).to_numpy(), indices=indices))

    def transform_splits(self, transformer):
        """
        Transform splits data
        """
        assert callable(transformer), 'formatter MUST be callable'

        for i, split in enumerate(self.splits):
            self.splits[i] = split._replace(np=transformer(split.np))

    def plot(self, title='', columns=None, n_ticks=15, grid=True, fontsize=6,  **kwargs):
        """
        Plot dataframe
        :param title: str title of plot
        :param columns: list columns to plot
        :param n_ticks: int number of ticks on the x axis
        :param grid: bool wether to display the grid
        :param fontsize: int font size for the axis values
        """
        self.df[columns or self.columns].plot(title=title, xticks=range(0, self.length, self.length // n_ticks), grid=grid, fontsize=fontsize, **kwargs)

    def plot_splits(self, columns=None, n_ticks=15, grid=True, fontsize=6, **kwargs):
        """
        Plot each of the splits
        :param columns: list columns to plot
        :param n_ticks: int number of ticks on the x axis
        :param grid: bool wether to display the grid
        :param fontsize: int font size for the axis values
        """
        for split in self.splits:
            split.df[columns or self.columns].plot(title=split.name, xticks=range(0, len(split.df), len(split.df) // n_ticks), grid=grid, fontsize=fontsize, **kwargs)

    def plot_splits_pca(self, alpha=0.2, s=2, **kwargs):
        """
        Plot 2 PCA components of splits
        """
        X = PCA(n_components=2).fit_transform(self.X)
        fig, ax = plt.subplots()

        # apply log scales
        if abs(X[:, 0].max() - X[:, 0].min()) > 1000:
            X[:, 0] -= X[:, 0].min() + 1
            ax.set_xscale('log')

        if abs(X[:, 1].max() - X[:, 1].min()) > 1000:
            X[:, 1] -= X[:, 1].min() + 1
            ax.set_yscale('log')

        ax.scatter(X[:, 0], X[:, 1], c=self.y, alpha=alpha, s=s, **kwargs)