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
        "y_true": y_true,
        "y_pred": y_pred,
        "score": metric(y_true, y_pred)
    }


class GridSearch:
    """
    Pipeline grid search
    """
    def __init__(self):
        """

        """
        # initialize paths to be a list with an empty path
        self.paths = [[]]

    @property
    def possibilities(self):
        """
        Better name for paths
        """
        return self.paths

    def then(self, steps):
        """
        Add more steps
        """
        if not isinstance(steps, list):
            steps = [steps]

        self.paths = [path + copy(steps) for path in self.paths]

        return self

    def one_of(self, paths):
        """
        Add branch
        """
        new_paths = []

        for path in paths:
            if path is None:
                path = []
            if not isinstance(path, list):
                path = [path]

            new_paths += [copy(existing_path) + copy(path) for existing_path in self.paths]

        self.paths = new_paths

        return self

    def optionally_one_of(self, paths):
        """
        Optionally add the given paths
        """
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

        return sorted([r for r in results if r is not None], key=lambda result: result["score"], reverse=True)
