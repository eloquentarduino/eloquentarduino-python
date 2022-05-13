from collections import Iterable
from itertools import product
from pprint import pprint
from tqdm import tqdm
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

    @property
    def compliant_results(self):
        """
        Get results that satisfy constraints
        """
        compliant = [self.test_result(result, return_result=True) for result in self.results]

        return [result for result in compliant if result is not None]

    def print_combinations(self):
        """
        Print all combinations
        """
        for i, combination in enumerate(self.combinations):
            print('#%3d' % (i + 1), combination)

    def print_results(self, n=999, resources=False, inference_time=False):
        """
        Print results
        :param n: int
        :param resources: bool
        :param inference_time: bool
        """
        for i, result in enumerate(self.compliant_results[:n]):
            print('#%3d Accuracy: %.2f | %s' % (i + 1, result.accuracy, str(result.hyperparameters)))

            if resources:
                print('Resources: ', end='')
                pprint(result.resources)

            if inference_time:
                print('Inference time: %.1f uS' % result.inference_time)

    def search(self, cv=3, project=None):
        """
        Perform search
        :param cv: int cross validation rounds
        :param project: Project
        :param show_progress: bool if True, a progress indicator is shown
        """
        self.results = []

        for combination in tqdm(list(self.combinations)):
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


