import random
from functools import reduce

import imblearn.under_sampling
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from eloquentarduino.utils.jinja import jinja
from eloquentarduino.ml.data.mixins.LoadsDatasetMixin import LoadsDatasetMixin
from eloquentarduino.ml.data.mixins.PlotsItselfMixin import PlotsItselfMixin
from eloquentarduino.ml.data.mixins.DropsTimeSeriesOutliersMixin import DropsTimeSeriesOutliersMixin


class Dataset(LoadsDatasetMixin, DropsTimeSeriesOutliersMixin, PlotsItselfMixin):
    """
    Abstraction of a dataset
    """
    def __init__(self, name, X, y, columns=None, classmap=None, test_validity=True):
        """
        :param name:
        :param X:
        :param y:
        :param columns: list
        :param classmap: dict
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
        self.classmap = classmap or {-1: 'UNLABELLED'}

    def __getitem__(self, item):
        """
        Access slice of data
        """
        return self.replace(X=self.X[item, :], y=self.y[item])

    def __len__(self):
        """
        Get length of dataset
        :return: int
        """
        return self.length

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
        Get number of samples
        """
        return self.num_samples

    @property
    def num_samples(self):
        """
        Get number of samples
        :return: int
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

    def update_classmap(self, classmap):
        """
        Update mapping from class id to class name
        :param classmap: dict
        :return: self
        """
        assert isinstance(classmap, dict), 'classmap MUST be a dict'
        self.classmap.update(classmap)

        return self

    def train_test_split(self, **kwargs):
        """
        Random train/test split
        @breaking-change 0.1.11 return datasets instead of raw arrays
        :return: tuple (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, **kwargs)

        return self.replace(X=X_train, y=y_train, name='%s train' % self.name), self.replace(X=X_test, y=y_test, name='%s test' % self.name)

    def drop_unlabelled(self):
        """
        Remove unlabelled samples
        @breaking-change 0.1.11 return a new Dataset instead of modifying in-place
        :return: Dataset
        """
        return self.mask(self.y >= 0)

    def label_where(self, label, mask_or_callable, label_name=None):
        """
        Set label where mask or the callable is True
        :param label: int|str idx of label
        :param mask_or_callable: numpy.ndarray|callable mask or callable that returns a mask
        :param label_name: str|None add label name to classmap
        """
        # if label is a string, convert it to idx
        if isinstance(label, str):
            label_name = label
            label = self._get_label_id(label)

        if callable(mask_or_callable):
            mask_or_callable = mask_or_callable(self.df)

        self.y = np.where(mask_or_callable, label, self.y)

        if label_name is not None:
            self.update_classmap({label: label_name})

        return self

    def label_samples(self, label, *ranges, **kwargs):
        """
        Add a label to a subset of the dataset
        :param label: str or int name of the given samples
        :param ranges: list of (start, end) tuples
        :return: self
        """
        if isinstance(label, str):
            label_id = self._get_label_id(label)
            self.classmap[label_id] = label
        else:
            label_id = label
            label_name = kwargs.get('label_name')

            if label_name:
                self.classmap[label_id] = label_name

        for start, end in ranges:
            self.y[start:end] = label_id

        return self

    def replace(self, X=None, y=None, columns=None, name=None, classmap=None):
        """
        Replace X and y
        :param X:
        :param y:
        """
        if X is None:
            X = self.X

        if y is None:
            y = self.y

        if columns is None:
            columns = self.columns

        if name is None:
            name = self.name

        if classmap is None:
            classmap = self.classmap

        return Dataset(name=name, X=X.copy(), y=y.copy(), columns=columns, classmap=classmap)

    def clone(self):
        """
        Clone dataset
        :return: Dataset
        """
        return self.replace()

    def select_columns(self, columns):
        """
        Only keep given columns
        :param columns: list of int or list of str
        :return: Dataset
        """
        column_indices = [i for i, column in enumerate(self.columns) if column in columns or i in columns]
        column_names = [column for i, column in enumerate(self.columns) if column in columns or i in columns]
        X = self.X[:, column_indices]

        return self.replace(X=X, columns=column_names)

    def shuffle(self, **kwargs):
        """
        Shuffle X and y
        @breaking-change 0.1.11 return new copy of self instead of in-place
        :return: Dataset
        """
        X, y = shuffle(self.X, self.y, **kwargs)

        return self.replace(X=X, y=y)

    def merge(self, other, *others):
        """
        Merge datasets with the same structure
        :param other: Dataset
        """
        assert isinstance(other, Dataset), 'you can only merge Datasets (%s given)' % str(type(other))
        assert self.num_features == other.num_features, 'you can only merge Datasets with the same number of features'

        X = np.vstack((self.X, other.X))
        y = np.concatenate((self.y, other.y))
        classmap = {**other.classmap, **self.classmap}
        merged = self.replace(X=X, y=y, classmap=classmap)

        for other in others:
            merged = merged.merge(other)

        return merged

    def mask(self, mask_or_callable):
        """
        Mask X and y
        @breaking-chage 0.1.11 return new dataset instead of in-place
        :param mask_or_callable: numpy.ndarray or callable
        :returns: Dataset
        """
        if callable(mask_or_callable):
            mask = mask_or_callable(self.df)
        else:
            mask = mask_or_callable

        return self.replace(X=self.X[mask], y=self.y[mask])

    def random(self, size=0):
        """
        Get random samples
        :param size: int number of samples to return
        :returns: Dataset
        """
        if size == 0:
            size = self.length

        idx = np.random.permutation(self.length)[:size]

        return self.replace(X=self.X[idx], y=self.y[idx])

    def intermix(self, chunk_size, *args, **kwargs):
        """
        @deprecated 0.1.11
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

        return self.replace(X=X, y=y)

    def take(self, size):
        """
        Take a subset of the dataset
        @deprecated 0.1.11
        :param size: int number of samples to keep
        """
        X, y = self.random(size)

        return self.replace(X=X, y=y)

    def keep_gaussian(self, multiplier=3):
        """
        Discard outliers based on variance
        @experimental
        @breaking-change 0.1.11 return new instance instead of in-place
        :param multiplier: float how many std to keep
        :returns: self
        """
        mean = self.X.mean(axis=0)
        std = self.X.std(axis=0)
        lower = mean - multiplier * std
        upper = mean + multiplier * std
        keep = np.all((self.X >= lower) & (self.X <= upper), axis=1)

        return self.mask(keep)

    def split(self, test=0, validation=0, return_empty=True, shuffle=True):
        """
        Split array into train, validation, test
        @deprecated 0.1.11
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

        return self.replace(X=X, y=y)

    def chunk(self, chunk_size, shuffle=False):
        """
        Chunk each class samples
        @deprecated 0.1.11
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

    def in_sequential_order(self):
        """
        Rearrange data so that all the samples for each class
        are on the same "block"
        @deprecated 0.1.11 same as sort_by_class()
        :return: Dataset
        """
        X = None
        y = None

        for class_idx in sorted(list(set(self.y))):
            Xi = self.X[self.y == class_idx]
            yi = np.ones(len(Xi)) * class_idx

            if X is None:
                X = Xi
                y = yi
            else:
                X = np.vstack((X, Xi))
                y = np.concatenate((y, yi))

        return self.replace(X=X, y=y)

    def train_test_split_sequential(self, test_size=0.33):
        """
        Split data into train/test keeping the time sequence from original data
        :param test_size: float in range 0-1
        :return: Dataset
        """
        X_train, X_test, y_train, y_test = None, None, None, None

        for class_idx in sorted(list(set(self.y))):
            Xi = self.X[self.y == class_idx]
            test_split = int(len(Xi) * test_size)
            Xi_train = Xi[:-test_split]
            Xi_test = Xi[-test_split:]
            yi_train = np.ones(len(Xi_train)) * class_idx
            yi_test = np.ones(len(Xi_test)) * class_idx

            if X_train is None:
                X_train = Xi_train
                X_test = Xi_test
                y_train = yi_train
                y_test = yi_test
            else:
                X_train = np.vstack((X_train, Xi_train))
                X_test = np.vstack((X_test, Xi_test))
                y_train = np.concatenate((y_train, yi_train))
                y_test = np.concatenate((y_test, yi_test))

        assert len(X_train) == len(y_train), 'something went wrong in training dataset'
        assert len(X_test) == len(y_test), 'something went wrong in testing dataset'

        return self.replace(X=X_train, y=y_train), self.replace(X=X_test, y=y_test)

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
        @breaking-change 0.1.11 return new instance instead of in-place
        :param min_width: int
        :param max_hop: int
        :param direction: str
        :param labels: list
        :returns: Dataset
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

        return self.replace(y=y)

    def expand_label(self, label, pad, min_width=0, direction='right'):
        """
        Expand given label to neighbor samples
        :param label: int or str
        :param pad: int
        :param min_width: int
        :param direction: str
        :returns: Dataset
        """
        distribution = self.y_segments()
        values = [v for v, s, l in distribution]
        starts = [s for v, s, l in distribution]
        lengths = [l for v, s, l in distribution]
        y = self.y.copy()

        if isinstance(label, str):
            label = self._get_label_id(label)

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

        return self.replace(y=y)

    def sort_by_class(self):
        """
        Sort samples by class.
        Useful if you have mixed samples and want to "compact" the samples from the same class.
        This will lose the "transient" dynamics, but should help improve the precision
        """
        indices = np.argsort(self.y)

        return self.replace(X=self.X[indices], y=self.y[indices])

    def port_classmap(self):
        """
        Return C code to translate class id to class label
        :return: str C code
        """
        return jinja('ml/utils/classmap.jinja', {'classmap': {k: v for k, v in self.classmap.items() if k >= 0}})

    def _get_label_id(self, label):
        """
        Get id for a label
        :param label: str
        """
        if isinstance(label, int):
            return label

        for lid, lab in self.classmap.items():
            if label == lab:
                return lid

        return max(self.classmap.keys()) + 1
