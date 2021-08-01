from collections import namedtuple, Iterable
from itertools import product
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from eloquentarduino.ml.data import Dataset
from eloquentarduino.ml.classification.abstract.GridSearch import GridSearch as GridSearchBase
from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier
from eloquentarduino.ml.classification.sklearn.gridsearch.GridSearchResult import GridSearchResult


class GridSearch(GridSearchBase):
    """
    Perform grid search parameter optimization on sklearn classifiers
    """
    def __init__(self, clf, dataset, only={}, also={}, exclude=[]):
        """
        :param clf: SklearnClassifier
        :param dataset: Dataset
        :param only: dict parameters to optimize only
        :param also: dict parameters to optimize in addition to defaults
        :param exclude: list parameters to exclude from search
        """
        assert isinstance(clf, SklearnClassifier), 'clf MUST be a SklearnClassifier'
        assert isinstance(dataset, Dataset), 'dataset MUST be a Dataset'

        super().__init__()
        self.clf = clf
        self.dataset = dataset
        self.only = only
        self.also = also
        self.exclude = exclude

    @property
    def combinations(self):
        """
        Get list of grid search combinations
        """
        defaults = self.clf.hyperparameters_grid(self.dataset.X)
        hyperparameters = defaults

        # only use supplied parameters for search
        # if parameter is None, use the default values
        if self.only:
            hyperparameters = {key: val if val is not None else defaults.get(key, []) for key, val in self.only.items()}
        elif self.also:
            hyperparameters.update(**self.also)

        for exclude in self.exclude:
            del hyperparameters[exclude]

        for key, values in hyperparameters.items():
            assert isinstance(values, Iterable), '%s values MUST be an iterable' % key

        for values in product(*hyperparameters.values()):
            yield {key: val for key, val in zip(hyperparameters.keys(), values)}

    def search(self, cv=3, project=None, show_progress=False):
        """
        Perform search
        :param cv: int cross validation rounds
        :param project: Project
        :param show_progress: bool if True, a progress indicator is shown
        """
        self.results = []

        for i, combination in enumerate(self.combinations):
            if show_progress:
                print(i if i % 20 == 0 else '.', end='')

            clf = clone(self.clf)
            clf.set_params(**combination)

            result = cross_validate(clf, self.dataset.X, self.dataset.y, cv=cv, return_estimator=True)
            best_idx = result['test_score'].argmax()
            accuracy = result['test_score'].mean()

            if accuracy > 0:
                clf = result['estimator'][best_idx]

                result = GridSearchResult(clf=clf, dataset=self.dataset, hyperparameters=combination, accuracy=accuracy, passes=False)
                result.passes, result.fail_reason = self.test_result(result, project=project)
                self.results.append(result)
            else:
                raise ValueError('Error evaluating accuracy for %s' % str(combination))

        self.results = sorted(self.results, key=lambda result: result.accuracy + (1 if result.passes else 0), reverse=True)

        return self.results

    def instantiate(self, i=0, **kwargs):
        """
        Instantiate result
        :param i: int
        :return: sklearn.Classifier
        """
        assert len(self.results) > 0, 'Unfitted'
        assert i < len(self.results), '%d is out of range'

        result = self.results[i]
        clf = result.clf

        clf.fit(self.dataset.X, self.dataset.y, **kwargs)

        return clf


