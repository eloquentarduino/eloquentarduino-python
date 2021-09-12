import warnings
from copy import copy

import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from eloquentarduino.ml.data.preprocessing.pipeline import Pipeline


def _evaluate_pipeline(pipeline, test, metric):
    """
    Compute score of pipeline on test set
    """
    try:
        y_pred, y_true = pipeline.fit().transform(test.X, test.y)
        y_pred = y_pred.flatten()
    except ValueError:
        return None

    return {
        "pipeline": pipeline,
        "score": metric(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred
    }


class GridSearch:
    """
    Pipeline grid search
    """
    def __init__(self):
        """

        """
        # initialize paths to be a list with an empty path
        self.paths = [None]
        self.results = []

    @property
    def possibilities(self):
        """
        Better name for paths
        """
        return self.paths

    def then(self, steps):
        """
        Add more steps
        :param steps: list
        """
        # allow NN grid search
        if hasattr(steps, "possibilities"):
            return self.one_of(steps.possibilities)

        self.paths = [self._copy(path) + self._copy(steps) for path in self.paths]

        return self

    def one_of(self, paths):
        """
        Add branch
        :param paths: list
        """
        new_paths = []

        # allow NN grid search
        if hasattr(paths, "possibilities"):
            paths = paths.possibilities

        for path in paths:
            new_paths += [self._copy(existing_path) + self._copy(path) for existing_path in self.paths]

        self.paths = new_paths

        return self

    def optionally_then(self, steps):
        """
        Optionally add the given steps
        :param steps: list
        """
        return self.one_of([None, steps])

    def optionally_one_of(self, paths):
        """
        Optionally choose between the given paths
        :param paths: list
        """
        # allow NN grid search
        if hasattr(paths, "possibilities"):
            paths = paths.possibilities

        return self.one_of([None] + paths)

    @np.errstate(all="ignore")
    def search(self, train, test, metric=accuracy_score):
        """
        Apply search to given train/test dataset
        :param train: Dataset
        :param test: Dataset
        :param metric: callable custom metric to sort results
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipelines = [Pipeline(name="Pipeline", dataset=train, steps=path) for path in self.paths]
            results = list(tqdm((_evaluate_pipeline(pipeline, test, metric) for pipeline in pipelines), total=len(self.possibilities)))

        self.results = sorted([r for r in results if r is not None], key=lambda result: result["score"], reverse=True)

        return self.results

    def sort_by(self, key):
        """
        Sort results by a custom function
        :param key: callable custom sorter
        :return: list of sorted results
        """
        return sorted(self.results, key=lambda result: key(result["y_true"], result["y_pred"]), reverse=True)

    def _copy(self, path):
        """
        Copy path
        :param path: list|any
        """
        if path is None:
            path = []
        elif not isinstance(path, list):
            path = [path]

        return [copy(step) for step in path]