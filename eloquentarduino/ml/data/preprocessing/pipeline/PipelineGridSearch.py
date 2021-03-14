from itertools import combinations
from math import floor
from collections import namedtuple
from collections.abc import Iterable

from eloquentarduino.ml.data.preprocessing.pipeline.Pipeline import Pipeline
from eloquentarduino.ml.data.preprocessing.pipeline.BoxCox import BoxCox
from eloquentarduino.ml.data.preprocessing.pipeline.Diff import Diff
from eloquentarduino.ml.data.preprocessing.pipeline.FFT import FFT
from eloquentarduino.ml.data.preprocessing.pipeline.MinMaxScaler import MinMaxScaler
from eloquentarduino.ml.data.preprocessing.pipeline.Norm import Norm
from eloquentarduino.ml.data.preprocessing.pipeline.PolynomialFeatures import PolynomialFeatures
from eloquentarduino.ml.data.preprocessing.pipeline.SelectKBest import SelectKBest
from eloquentarduino.ml.data.preprocessing.pipeline.StandardScaler import StandardScaler
from eloquentarduino.ml.data.preprocessing.pipeline.Window import Window
from eloquentarduino.ml.data.preprocessing.pipeline.YeoJohnson import YeoJohnson


class PipelineGridSearch:
    """
    Perform a naive grid search to find the best pipeline for the dataset
    """
    def __init__(self, dataset, is_time_series=False, duration=None, global_scaling=False, feature_selection=False):
        if is_time_series:
            assert duration > 0, 'if is_time_series=True, duration MUST be set'

        self.dataset = dataset
        self.is_time_series = is_time_series
        self.duration = duration
        self.global_scaling = global_scaling
        self.feature_selection = feature_selection

    def naive_search(self, clf, max_steps=3, cv=3, always_confirm=False, show_progress=False):
        """
        Perform a naive search for the optimal pipeline
        :param clf:
        :param max_steps: int max number of steps for a pipeline
        :param cv: int cross validation splits
        :param always_confirm: bool if True, the function doesn't ask for confirmation
        :param show_progress: bool if True, a progress indicator is shown
        """
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
                results.append(Result(pipeline=pipeline, accuracy=pipeline.score(clf, cv=cv, return_average_accuracy=True)))
            except ValueError:
                pass
            except AssertionError:
                pass

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

        if self.feature_selection:
            steps += [
                SelectKBest(name='1/4 K best', k=self.dataset.num_features // 4),
                SelectKBest(name='1/2 K best', k=self.dataset.num_features // 2),
            ]

        final_steps = []

        if self.is_time_series:
            final_steps = [
                [Window(length=self.duration)],
                [Window(name='Window for FFT', length=self.duration), FFT(num_features=1/self.duration)]
            ]

        for n in range(1, max_steps + 1):
            for combination in combinations(steps, n):
                if len(final_steps) > 0:
                    for final_step in final_steps:
                        yield list(combination) + final_step
                else:
                    yield list(combination)
