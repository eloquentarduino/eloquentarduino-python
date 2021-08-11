from itertools import combinations
from math import floor
from collections import namedtuple
from collections.abc import Iterable
from copy import copy

from eloquentarduino.ml.data.preprocessing.pipeline.Pipeline import Pipeline
from eloquentarduino.ml.data.preprocessing.pipeline.BoxCox import BoxCox
from eloquentarduino.ml.data.preprocessing.pipeline.Diff import Diff
from eloquentarduino.ml.data.preprocessing.pipeline.FFT import FFT
from eloquentarduino.ml.data.preprocessing.pipeline.MinMaxScaler import MinMaxScaler
from eloquentarduino.ml.data.preprocessing.pipeline.Norm import Norm
from eloquentarduino.ml.data.preprocessing.pipeline.PolynomialFeatures import PolynomialFeatures
from eloquentarduino.ml.data.preprocessing.pipeline.SelectKBest import SelectKBest
from eloquentarduino.ml.data.preprocessing.pipeline.StandardScaler import StandardScaler
from eloquentarduino.ml.data.preprocessing.pipeline.StatMoments import StatMoments
from eloquentarduino.ml.data.preprocessing.pipeline.Window import Window
from eloquentarduino.ml.data.preprocessing.pipeline.YeoJohnson import YeoJohnson


class PipelineGridSearch:
    """
    Perform a naive grid search to find the best pipeline for the dataset
    @deprecated
    """
    def __init__(self, dataset, is_time_series=False, duration=None, global_scaling=False, feature_selection=False):
        if is_time_series:
            assert duration > 0, 'if is_time_series=True, duration MUST be set'

        self.dataset = dataset
        self.is_time_series = is_time_series
        self.duration = duration
        self.global_scaling = global_scaling
        self.feature_selection = feature_selection

    def naive_search(self, clf, max_steps=3, cv=3, always_confirm=False, show_progress=False, test=None):
        """
        Perform a naive search for the optimal pipeline
        :param clf:
        :param max_steps: int max number of steps for a pipeline
        :param cv: int cross validation splits
        :param always_confirm: bool if True, the function doesn't ask for confirmation
        :param show_progress: bool if True, a progress indicator is shown
        :param test: tuple (X_test, y_test)
        """
        assert test is None or isinstance(test, tuple) and len(test) == 2, 'test MUST be None or a (X_test, y_test) tuple'

        Result = namedtuple('Result', 'pipeline accuracy')
        results = []
        combs = list(self.enumerate(max_steps))

        if not always_confirm and input('%d combinations will be tested: do you want to proceed? [y/n] ' % len(combs)).lower() != 'y':
            return

        for i, steps in enumerate(self.enumerate(max_steps)):
            if show_progress:
                print(i if i % 20 == 0 else '.', end='')
            try:
                pipeline = Pipeline('PipelineGridSearch', self.dataset, steps=steps)
                pipeline.fit()

                # estimate accuracy via cross-validation
                if test is None:
                    accuracy = pipeline.score(clf, cv=cv, return_average_accuracy=True)
                else:
                    # use provided test set
                    X_test, y_test = test
                    accuracy = clf.fit(pipeline.X, pipeline.y).score(*pipeline.transform(X_test, y_test))

                results.append(Result(pipeline=pipeline, accuracy=accuracy))
            except ValueError as ex:
                print('ValueError', ex)
            except IndexError as ex:
                print('Index error', ex)
            except AssertionError as ex:
                print('Assertion error', ex)

        return sorted(results, key=lambda result: -result.accuracy)

    def enumerate(self, max_steps):
        """
        Enumerate every possible combination of steps
        :param max_steps: int
        """
        steps = [
            MinMaxScaler(),
            StandardScaler(),
            Norm(name='L1-Norm', norm='l1'),
            Norm(name='L2-Norm', norm='l2'),
            Norm(name='LInf-Norm', norm='inf'),
            BoxCox(),
            YeoJohnson(),
            Diff(),
            PolynomialFeatures()
        ]

        if self.global_scaling:
            steps += [
                MinMaxScaler(name='MinMaxScalerGlobal', num_features=0),
                StandardScaler(name='StandardScalerGlobal', num_features=0)
            ]

        final_steps = []

        if self.is_time_series:
            final_steps = [
                [Window(length=self.duration)],
                [Window(name='Window for FFT', length=self.duration), FFT(num_features=1/self.duration)],
                [Window(name='Window for FFT', length=self.duration), StatMoments(num_features=1/self.duration)]
            ]

        # #todo
        #if self.feature_selection:
        #    final_steps += [
        #        SelectKBest(name='1/4 K best', k=self.dataset.num_features // 4),
        #        SelectKBest(name='1/2 K best', k=self.dataset.num_features // 2),
        #    ]

        for n in range(1, max_steps + 1):
            for combination in combinations(steps, n):
                if len(final_steps) > 0:
                    for final_step in final_steps:
                        yield list(combination) + final_step
                else:
                    yield list(combination)
