from sklearn.base import clone
from itertools import combinations
from eloquentarduino.ml.data import Pipeline
from sklearn.model_selection import train_test_split
from math import floor
from collections.abc import Iterable


class PipelineGridSearch:
    def __init__(self, clf, X, y, test_size=0.3):
        self.clf = clf
        self.X = X
        self.y = y
        self.test_size = test_size

    def naive_search(self, feature_selection=True, return_all=False, verbose=True):
        """Perform a naive search for the optimal pipeline"""
        results = []
        for combination in self._generate_combinations(feature_selection=feature_selection):
            try:
                pipeline = Pipeline(self.X, self.y)
                for step in combination:
                    step(pipeline)

                X_train, X_test, y_train, y_test = train_test_split(pipeline.Xt, pipeline.y, test_size=self.test_size)
                clf = clone(self.clf)
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                if verbose:
                    print('Score with %d steps: %.2f' % (len(combination), score))
                results.append((pipeline, score))
            except ValueError as err:
                if verbose:
                    print('Bad combination', err)
            except AssertionError as err:
                if verbose:
                    print('Bad combination', err)

        results = sorted(results, key=lambda r: r[1], reverse=True)
        if return_all:
            return results
        # pick one with highest accuracy
        return results[0]

    def _generate_combinations(self, feature_selection):
        available_steps = [
            # lambda p: p.polynomial_features(),
            lambda p: p.standardize(),
            lambda p: p.standardize(featurewise=True),
            lambda p: p.normalize(),
            lambda p: p.normalize(featurewise=True),
            lambda p: p.fft()
        ]
        # if feature_selection=True, try a naive combination of k
        if isinstance(feature_selection, bool) and feature_selection:
            available_steps += [
                lambda p: p.select_kbest(k=5),
                lambda p: p.select_kbest(k=20),
                lambda p: p.select_kbest(k=floor(self.X.shape[1] / 2)),
                lambda p: p.select_kbest(k=self.X.shape[1]),
            ]
        # if feature_selection is a list, it's a list of ks
        elif isinstance(feature_selection, Iterable):
            available_steps += [lambda p: p.select_kbest(k=k) for k in feature_selection]

        # generate each possible combination
        for n in range(1, len(available_steps) + 1):
            combinations_of_n_elements = combinations(available_steps, n)
            for combination in combinations_of_n_elements:
                yield combination