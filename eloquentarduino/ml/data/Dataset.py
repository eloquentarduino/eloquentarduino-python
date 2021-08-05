import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from os.path import basename, splitext, sep
from glob import glob
from functools import reduce
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
import imblearn.under_sampling


class Dataset:
    """
    Abstraction of a dataset
    """
    @staticmethod
    def Iris():
        """
        Create the Iris dataset
        """
        return Dataset('Iris', *load_iris(return_X_y=True))

    @staticmethod
    def MNIST_Tensorflow():
        """
        Create the MNIST dataset formatted for Tensorflow
        """
        X, y = load_digits(return_X_y=True)

        return Dataset('MNIST Tf', np.expand_dims(X.reshape((-1, 8, 8)), -1), y, test_validity=False)

    @staticmethod
    def read_csv(filename, name=None, label_column=None, columns=None, **kwargs):
        """
        Create dataset from csv file
        """
        if name is None:
            name = splitext(basename(filename))[0]

        assert len(name) > 0, 'dataset name CANNOT be empty'

        df = pd.read_csv(filename, **kwargs)

        # select given columns
        if columns is not None:
            df = df[columns]

        # use label column or set y to -1
        if label_column is None:
            X = df.to_numpy()
            y = -np.ones(len(df))
        else:
            columns = [column for column in df.columns if column != label_column]
            X = df[columns].to_numpy()
            y = df[label_column].to_numpy()

        return Dataset(name, X, y, columns=columns)

    @staticmethod
    def read_folder(folder, file_pattern='*.csv', delimiter=',', skiprows=0, slice=None):
        """
        Load all files from a folder
        :param folder: str
        :param file_pattern: str pattern for glob()
        :param delimiter: str
        :param skiprows: int
        :param slice: tuple
        """
        X, y = None, None
        labels = []

        for class_idx, filename in enumerate(sorted(glob("%s/%s" % (folder, file_pattern)))):
            label = splitext(basename(filename))[0]
            try:
                Xi = np.loadtxt(filename, dtype=np.float, delimiter=delimiter, skiprows=skiprows)
            except ValueError:
                raise ValueError('Cannot read file %s' % filename)

            if slice is not None:
                start, end = slice
                Xi = Xi[start:end]

            yi = np.ones(len(Xi)) * class_idx
            labels.append((label, len(Xi)))

            if X is None:
                X = Xi
                y = yi
            else:
                X = np.vstack((X, Xi))
                y = np.concatenate((y, yi))

        assert X is not None, '%s is empty' % folder

        name = [segment for segment in folder.split(sep) if len(segment)][-1]
        dataset = Dataset(name, X, y)
        offset = 0

        for label, length in labels:
            dataset.label_samples(label, (offset, offset + length))
            offset += length

        return dataset

    def __init__(self, name, X, y, columns=None, test_validity=True):
        """
        :param name:
        :param X:
        :param y:
        :param columns:
        :param test_validity: bool
        """
        self.name = name

        if test_validity:
            try:
                valid_rows = ~np.isnan(X).any(axis=1)
            except TypeError:
                valid_rows = slice(0, 999999)

            self.X = X[valid_rows]
            self.y = np.asarray(y)[valid_rows].astype(np.int)
        else:
            self.X = X
            self.y = y

        self.columns = columns
        self.classmap = {-1: 'UNLABELLED'}

    @property
    def y_categorical(self):
        """
        Convert y to one-hot
        :return: ndarray
        """
        return OneHotEncoder(handle_unknown='ignore').fit_transform(self.y.reshape(-1, 1)).toarray()

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

    @property
    def shape(self):
        """
        Get X shape
        """
        return self.X.shape

    @property
    def df(self):
        """
        Convert dataset to pd.DataFrame
        """
        columns = self.columns if self.columns else ['f%d' % i for i in range(self.X.shape[1])]
        columns = [column for column in columns if columns != 'y']

        y = self.y.reshape((-1, 1))

        return pd.DataFrame(np.hstack((self.X, y)), columns=columns + ['y'])

    @property
    def class_labels(self):
        """
        Get labels of classes
        """
        return [label for idx, label in self.classmap.items() if idx >= 0]

    @property
    def class_distribution(self):
        """
        Get dict of {label: number of samples}
        """
        return {label: (self.y == label).sum() for label in sorted(list(set(self.y)))}

    def train_test_split(self, **kwargs):
        """
        Train/test split
        """
        return train_test_split(self.X, self.y, **kwargs)

    def drop_unlabelled(self):
        """
        Remove unlabelled samples
        """
        idx = (self.y == -1)
        self.X = self.X[~idx]
        self.y = self.y[~idx]

    def label_samples(self, label, *ranges):
        """
        Add a label to a subset of the dataset
        :param label: str name of the given samples
        :param ranges: list of (start, end) tuples
        """
        label_id = self._get_label_id(label)
        self.classmap[label_id] = label

        for start, end in ranges:
            self.y[start:end] = label_id

    def replace(self, X=None, y=None, columns=None):
        """
        Replace X and y
        :param X:
        :param y:
        """
        if X is None:
            X = self.X

        if y is None:
            y = self.y

        return Dataset(name=self.name, X=X.copy(), y=y.copy(), columns=columns)

    def shuffle(self, **kwargs):
        """
        Shuffle X and y
        :return: Dataset
        """
        self.X, self.y = shuffle(self.X, self.y, **kwargs)

        return self

    def merge(self, other, *args):
        """
        Merge datasets with the same structure
        :param other: Dataset
        """
        assert isinstance(other, Dataset), 'you can only merge Datasets'
        assert self.num_features == other.num_features, 'you can only merge Datasets with the same number of features'

        self.X = np.vstack((self.X, other.X))
        self.y = np.concatenate((self.y, other.y))

        for other in args:
            self.merge(other)

        return self

    def mask(self, mask):
        """
        Mask X and y
        """
        self.X = self.X[mask]
        self.y = self.y[mask]

    def random(self, size=0):
        """
        Get random samples
        :param size: int number of samples to return
        """
        if size == 0:
            size = self.length

        idx = np.random.permutation(self.length)[:size]
        return self.X[idx], self.y[idx]

    def intermix(self, chunk_size, *args, **kwargs):
        """
        @deprecated
        """
        return self.mix(chunk_size, *args, **kwargs)

    def mix(self, chunk_size):
        """
        Alternate samples in sizes of chunk_size based on the y values
        :param chunk_size: int
        """
        X_chunks = []
        y_chunks = []

        for idx in range(self.num_classes):
            class_mask = self.y == idx
            class_samples = self.X[class_mask]
            xi = np.array_split(class_samples, len(class_samples) // chunk_size)
            yi = [np.ones(len(xij), dtype=np.uint8) * idx for xij in xi]
            X_chunks += xi
            y_chunks += yi

        chunks = list(zip(X_chunks, y_chunks))
        random.shuffle(chunks)
        X_chunks, y_chunks = zip(*chunks)

        X = np.vstack(X_chunks)
        y = np.concatenate(y_chunks)

        return Dataset(self.name, X, y)

    def take(self, size):
        """
        Take a subset of the dataset
        """
        return Dataset(self.name, *self.random(size))

    def keep_gaussian(self, multiplier=3):
        """
        Discard outliers based on variance
        """
        mean = self.X.mean(axis=0)
        std = self.X.std(axis=0)
        lower = mean - multiplier * std
        upper = mean + multiplier * std
        keep = np.all((self.X >= lower) & (self.X <= upper), axis=1)

        self.X = self.X[keep]
        self.y = self.y[keep]

    def split(self, test=0, validation=0, return_empty=True, shuffle=True):
        """
        Split array into train, validation, test
        :param test: float test size percent
        :param validation: float validation size percent
        :param return_empty: bool if empty splits should be returned
        :param shuffle: bool if dataset should be shuffled before splitting
        """
        assert test > 0 or validation > 0, 'either test or validation MUST be greater than 0'
        assert test + validation < 1, 'test + validation MUST be less than 1'
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

    def balance(self, minority_ratio=1, chunk_size=None, kmeans=False, nearmiss=False, tomek=False, random_state=0, **kwargs):
        """
        Balance unbalanced classes
        :param minority_ratio: float ratio between majority classes and minority class
        :param chunk_size: int
        """
        undersampler = None

        if kmeans:
            undersampler = imblearn.under_sampling.ClusterCentroids(sampling_strategy=minority_ratio, random_state=random_state, **kwargs)
        elif nearmiss:
            undersampler = imblearn.under_sampling.NearMiss(sampling_strategy=minority_ratio, **kwargs)
        elif tomek:
            undersampler = imblearn.under_sampling.TomekLinks(**kwargs)

        if undersampler is not None:
            X, y = undersampler.fit_resample(self.X, self.y)
        else:
            assert chunk_size > 1, 'chunk_size MUST be greater than 1'

            # naive, "sequential" undersampling (keeps time series consistency)
            class_distribution = self.class_distribution
            minority_samples = min(class_distribution.values())
            # here ratio is a multiplier of minority_samples
            ratio = 1 / minority_ratio
            class_chunks = self.chunk(chunk_size=chunk_size, shuffle=True)
            max_chunks = int(minority_samples * ratio // chunk_size)
            all_chunks = []

            for label, chunks in class_chunks.items():
                all_chunks += [(label, chunk) for chunk in chunks[:max_chunks]]

            random.shuffle(all_chunks)
            X = np.vstack([chunk for label, chunk in all_chunks])
            y = np.concatenate([np.ones(len(chunk)) * label for label, chunk in all_chunks])

        return Dataset(self.name, X, y)

    def chunk(self, chunk_size, shuffle=False):
        """
        Chunk each class samples
        :param chunk_size: int
        :param shuffle: bool
        """
        chunked = {}

        for yi in sorted(set(self.y)):
            mask = self.y == yi
            Xi = self.X[mask]
            num_chunks = int(len(Xi) // chunk_size)
            chunks = np.array_split(Xi, num_chunks)

            if shuffle:
                indices = np.arange(len(chunks))
                np.random.shuffle(indices)
                chunks = [chunks[i] for i in indices]

            chunked[int(yi)] = chunks

        return chunked

    def train_test_split_chunks(self, chunk_size, test_size=0.33):
        """
        Split data in train/test considering chunks of data
        :param chunk_size: int
        :param test_size: float
        """
        class_chunks = self.chunk(chunk_size, shuffle=True)
        splits = [int(len(chunks) * (1 - test_size)) for chunks in class_chunks.values()]
        train_chunks = [(label, chunk) for (label, chunks), split in zip(class_chunks.items(), splits) for chunk in chunks[:split]]
        test_chunks = [(label, chunk) for (label, chunks), split in zip(class_chunks.items(), splits) for chunk in chunks[split:]]
        random.shuffle(train_chunks)
        random.shuffle(test_chunks)
        X_train = np.vstack([chunk for label, chunk in train_chunks])
        X_test = np.vstack([chunk for label, chunk in test_chunks])
        y_train = np.concatenate([np.ones(len(chunk)) * label for label, chunk in train_chunks])
        y_test = np.concatenate([np.ones(len(chunk)) * label for label, chunk in test_chunks])

        return Dataset('%s train' % self.name, X_train, y_train), Dataset('%s test' % self.name, X_test, y_test)

    def y_segments(self):
        y = self.y.copy().astype(int)
        loc_run_start = np.empty(len(y), dtype=bool)
        loc_run_start[0] = True
        np.not_equal(y[:-1], y[1:], out=loc_run_start[1:])
        starts = np.nonzero(loc_run_start)[0]
        lengths = np.diff(np.append(starts, len(y)))
        values = y[loc_run_start]

        return list(zip(values, starts, lengths))

    def fill_holes(self, min_width, max_hop, direction='right', labels=None):
        """
        Fill holes in label column
        """
        distribution = self.y_segments()
        values = [v for v, s, l in distribution]
        starts = [s for v, s, l in distribution]
        lengths = [l for v, s, l in distribution]
        y = self.y.copy()

        if direction == 'right':
            for v_start, v_end, width, hop, start in zip(values[:-1], values[2:], lengths[:-1],
                                                         lengths[1:], starts[:-1]):
                if (
                        labels is not None and v_start not in labels) or v_start != v_end or width < min_width or hop > max_hop:
                    continue
                y[start + width:start + width + hop] = v_start

        elif direction == 'center':
            for v_start, v_end, width1, hop, width2, start in zip(values[:-1], values[2:], lengths[:-1],
                                                                  lengths[1:], lengths[2:], starts[:-1]):
                if (
                        labels is not None and v_start not in labels) or v_start != v_end or width1 + width2 < min_width or hop > max_hop:
                    continue
                y[start + width1:start + width1 + hop] = v_start

        self.y = y

    def expand_label(self, label, pad, min_width=0, direction='right'):
        """
        Expand given label to neighbor samples
        """
        distribution = self.y_segments()
        values = [v for v, s, l in distribution]
        starts = [s for v, s, l in distribution]
        lengths = [l for v, s, l in distribution]
        y = self.y.copy()

        if direction == 'right':
            for v, width, start in zip(values[:-1], lengths[:-1], starts[:-1]):
                if v != label or width < min_width:
                    continue
                y[start + width: start + width + pad] = label

        if direction == 'left':
            for v, width, start in zip(values[1:], lengths[1:], starts[1:]):
                if v != label or width < min_width:
                    continue
                y[start - pad: start] = label

        elif direction == 'center':
            for v, width, start in zip(values[:-1], lengths[:-1], starts[:-1]):
                if v != label or width < min_width:
                    continue
                y[start - pad: start + width + pad] = label

        self.y = y

    def plot(self, title='', columns=None, n_ticks=15, grid=True, fontsize=6, bg_alpha=0.2, once_every=1, palette=None, y_pred=None, **kwargs):
        """
        Plot dataframe
        :param title: str title of plot
        :param columns: list columns to plot
        :param n_ticks: int number of ticks on the x axis
        :param grid: bool wether to display the grid
        :param fontsize: int font size for the axis values
        :param bg_alpha: float alpha of classes' background color
        :param once_every: int limit the number of samples to draw
        """
        plt.figure()
        plot_columns = [c for c in (columns or list(self.df.columns)) if c != 'y']
        df = pd.DataFrame(self.df[plot_columns].iloc[::once_every].to_numpy(), columns=plot_columns)
        length = len(df)

        df.plot(title=title, xticks=range(0, length, length // n_ticks), grid=grid, fontsize=fontsize, rot=70, **kwargs)

        # highlight labels
        y = self.y[::once_every]
        loc_run_start = np.empty(len(y), dtype=bool)
        loc_run_start[0] = True
        np.not_equal(y[:-1], y[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]
        run_lengths = np.diff(np.append(run_starts, len(y)))
        run_values = y[loc_run_start]
        palette = [c for c in (palette or mcolors.TABLEAU_COLORS.values())]

        if self.y.max() >= len(palette):
            print('[WARN] too many classes for the current palette')

        for v, s, l in zip(run_values, run_starts, run_lengths):
            plt.axvspan(s, s + l, color=palette[v % len(palette)], alpha=bg_alpha)

        # plot y_test markers
        if y_pred is not None:
            hop = len(self.y) // len(y_pred)
            zero = self.X.min()
            # markers = 'ovsP*+x1<p'

            for i, yi in enumerate(set(y_pred)):
                scale = 1 - i * 0.025 if zero > 0 else 1 + i * 0.025
                xs = np.argwhere(y_pred == yi).flatten() * hop + hop
                ys = np.ones(len(xs)) * zero * scale
                plt.scatter(xs, ys, marker='.', c=palette[i % len(palette)], s=2)

    def _get_label_id(self, label):
        """
        Get id for a label
        :param label: str
        """
        for lid, lab in self.classmap.items():
            if label == lab:
                return lid

        return max(self.classmap.keys()) + 1
