import numpy as np
from time import perf_counter
from copy import copy
from os.path import sep, basename, splitext
from glob import glob
from sklearn.base import clone
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from pandas import DataFrame
from eloquentarduino.ml.metrics.plot import Barplot, ConfusionMatrix


def benchmark_time(f):
    """
    Benchmark how much it takes to run the function
    :param f:
    :return:
    """
    start = perf_counter()
    f()
    return perf_counter() - start


class Dataset:
    def __init__(self, name, X, y):
        self.fullname = name
        self.name = name
        self.X = X
        self.y = y
        self.pipeline = None

        if len(self.name) > 10:
            self.name = '%s...' % self.name[:8]

    @property
    def shape(self):
        """
        Get shape of data
        :return:
        """
        return self.X.shape


class ClassifierBenchmarkResult(dict):
    """
    The result of a benchmark
    """
    def __init__(self, label, clf, dataset, X_train, X_test, y_train, y_test):
        self.label = label
        self.dataset = dataset.name
        self.shape = dataset.shape
        self.training_time = benchmark_time(lambda: clf.fit(X_train, y_train))

        y_pred = clf.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred, average='micro')
        self.recall = recall_score(y_test, y_pred, average='micro')
        self.f1 = f1_score(y_test, y_pred, average='micro')
        self.confusion_matrix = confusion_matrix(y_test, y_pred, normalize='true')

    def to_dict(self):
        """
        Convert to dict
        :return:
        """
        return self.__dict__


class ClassifierBenchmark:
    """
    Benchmark classifier on the given datasets
    """
    def __init__(self, datasets=['toy'], plots=None, clf=None, **kwargs):
        self.datasets = list(self.load_datasets(datasets))
        self.plots = plots
        self.clf = clf
        self.results = []

        if clf is not None:
            self.run(**kwargs)

    @property
    def df(self):
        """
        Convert results to DataFramce
        :return:
        """
        data = [r.to_dict() for r in self.results]
        return DataFrame(data, columns=['label', 'dataset', 'shape', 'training_time', 'accuracy', 'precision', 'recall', 'f1'])

    def __call__(self, *args, **kwargs):
        clone = copy(self)
        clone.results = []

        return clone

    def preprocess(self, dataset_name, pipeline):
        """
        Add ml pipeline to datasets
        :param dataset_name:
        :param pipeline:
        :return:
        """
        for dataset in self.datasets:
            if dataset_name is None or dataset.fullname == dataset_name:
                dataset.pipeline = pipeline(dataset) if callable(pipeline) else copy(pipeline)

        return self

    def run(self, label, clf_proto, test_size=0.3, random_state=0, **kwargs):
        """
        Run the benchmark on the given datasets
        :param label:
        :param clf_proto:
        :param test_size:
        :param random_state:
        :return:
        """
        for dataset in self.datasets:
            clf = clone(clf_proto)

            # set custom params for classifier
            params = {k: v(dataset) if callable(v) else v for k, v in kwargs.items()}
            clf.set_params(**params)

            X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=test_size, random_state=random_state)

            if dataset.pipeline is not None:
                pipeline = dataset.pipeline.fit(X_train, y_train)
                X_train = pipeline.transform(X_train)
                X_test = pipeline.transform(X_test)

            self.results.append(ClassifierBenchmarkResult(label, clf, dataset, X_train, X_test, y_train, y_test))

        return self

    def load_datasets(self, datasets):
        for d in datasets:
            # load sklearn toy datasets
            if d == 'toy':
                yield Dataset('Iris', *load_iris(return_X_y=True))
                yield Dataset('Breast cancer', *load_breast_cancer(return_X_y=True))
                yield Dataset('Wine', *load_wine(return_X_y=True))
                yield Dataset('Digits', *load_digits(return_X_y=True))
            # a folder
            elif sep in d:
                # append trailing /
                if not d.endswith(sep):
                    d += sep
                # add glob pattern
                if '*' not in d:
                    d += '*.csv'
                for filename in glob(d, recursive=True):
                    data = np.loadtxt(filename, delimiter=',')
                    yield Dataset(splitext(basename(filename))[0], data[:, :-1], data[:, -1])

    def plot(self, compare=None, plots=None, sort=False):
        """
        Plot results
        :param sort:
        :return:
        """
        if plots is None:
            plots = self.plots

        for plot in plots:
            if plot == 'training_time':
                Barplot(self.df, x='dataset', y='training_time', compare=compare, sort=sort, factor='label')
            elif plot == 'accuracy':
                Barplot(self.df, x='dataset', y='accuracy', compare=compare, sort=sort, factor='label')
            elif plot == 'confusion_matrix':
                results_compare = compare.results if compare is not None else []
                for result, result_compare in zip(self.results, results_compare):
                    compare_confusion_matrix = result_compare.confusion_matrix if result_compare is not None else None
                    ConfusionMatrix(result.confusion_matrix, compare=compare_confusion_matrix).plot(label=result.dataset)
