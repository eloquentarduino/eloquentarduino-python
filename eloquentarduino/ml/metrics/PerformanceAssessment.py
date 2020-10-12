from collections import namedtuple
from copy import copy
from sklearn.base import clone
from sklearn.datasets import *
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from eloquentarduino.ml.metrics.PerformanceAssessmentResultsPlotter import PerformanceAssessmentResultsPlotter


Dataset = namedtuple('Dataset', 'name loader')
PerformanceAssessmentResult = namedtuple('PerformanceAssessmentResult', 'dataset shape clf accuracy precision recall f1 confusion_matrix')


class PerformanceAssessment:
    """Compute metrics about a classifier on various datasets"""
    def __init__(self):
        self._datasets = []
        self._results = []
        self._baseline = None
        self.add_dataset('iris', load_iris)
        self.add_dataset('breast cancer', load_breast_cancer)
        self.add_dataset('wine', load_wine)
        self.add_dataset('digits', load_digits)

    @property
    def results(self):
        """
        Getter for _results
        :return:
        """
        return self._results

    @property
    def baseline(self):
        """
        Getter for _results
        :return:
        """
        return self._baseline.results if self._baseline is not None else None

    @property
    def plotter(self):
        """
        Get plotter for self
        :return: PerformanceAssessmentResultsPlotter
        """
        return PerformanceAssessmentResultsPlotter(self)

    def unload_dataset(self, name=None):
        """
        Unload one or all of the datasets
        :param name: name of the dataset to unload or None to unload all
        :return: self
        """
        if name is None:
            self._datasets = []
        else:
            self._datasets = [dataset for dataset in self._datasets if dataset.name != name]
        return self

    def add_dataset(self, name, loader):
        """
        Add dataset to test
        :param name: name of the dataset
        :param loader: function to load the dataset
        :return: self
        """
        assert isinstance(name, str), 'name MUST be a string'
        assert callable(loader), 'loader MUST be callable'
        self._datasets.append(Dataset(name=name, loader=loader))
        return self

    def set_baseline(self, results):
        """
        Set results baseline
        :param results: baseline
        :return: clone of self
        """
        self._baseline = results
        return self

    def run(self, clf, pipeline=None, test_size=0.3, **kwargs):
        """
        Run performance assessment on given classifier
        :param clf: ML classifier
        :param pipeline: scikit-learn pipeline to preprocess data
        :param test_size: for train_test_split
        :param kwargs: customize classifier for each dataset. If value is a function, it receives (dataset.name, X, y) arguments
        :return: self
        """
        self._results = []

        for dataset in self._datasets:
            X, y = dataset.loader(return_X_y=True)
            if pipeline is not None:
                X = pipeline.transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            # create clf
            dataset_clf = clone(clf)
            dataset_params = {k: v(dataset.name, X, y) if callable(v) else v for k, v in kwargs.items()}
            dataset_clf.set_params(**dataset_params)

            y_pred = dataset_clf.fit(X_train, y_train).predict(X_test)

            self._results.append(PerformanceAssessmentResult(
                dataset=dataset.name,
                clf=clf,
                shape=X_test.shape,
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred, average=None),
                recall=recall_score(y_test, y_pred, average=None),
                f1=f1_score(y_test, y_pred, average=None),
                confusion_matrix=confusion_matrix(y_test, y_pred, normalize='true')
            ))

        return self
