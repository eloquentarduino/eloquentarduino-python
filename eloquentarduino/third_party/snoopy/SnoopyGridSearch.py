import numpy as np
from copy import copy
from sklearn.metrics import f1_score, balanced_accuracy_score
from eloquentarduino.ml.data import Dataset
from eloquentarduino.ml.data.preprocessing.pipeline import *
from eloquentarduino.ml.classification.sklearn import *
from eloquentarduino.third_party.snoopy.Voting import Voting
from eloquentarduino.third_party.snoopy.SmoothVoting import SmoothVoting


class SnoopyGridSearch:
    """
    Perform grid search on well-known configurations
    """
    def __init__(self, clf=None, pipeline=None, short_votes=None, long_votes=None, quorum=None):
        self.clf = clf
        self.pipeline = pipeline
        self.short_votes = short_votes or [1, 5, 10, 15]
        self.long_votes = long_votes or [1, 3, 5, 10]
        self.quorum = quorum or [0.5, 0.7, 0.85]
        self.results = []

    def search_window_size(self, train, test, length=None, shift=None):
        """
        :param length: int
        :param shift:int
        """
        assert self.pipeline is not None, 'you MUST set a pipeline'
        assert len([step for step in self.pipeline.steps if isinstance(step, Window)]) > 0, 'pipeline MUST contain a Window step'
        assert self.clf is not None, 'you MUST set a classifier'

        results = []

        # default lengths
        if length is None:
            length = [12, 25, 50]

        # default shifts
        if shift is None:
            shift = [1, 2, 6, 12, 25]

        for l in length:
            for s in shift:
                steps = [copy(step) for step in self.pipeline.steps]

                # overwrite window length and shift
                for step in steps:
                    if isinstance(step, Window):
                        step.name = 'Window'
                        step.length = l
                        step.shift = s
                        break

                results.append(self.score(pipeline=Pipeline('Pipeline', train, steps=steps).fit(), test=test))

        self.results = sorted(results, key=lambda result: result['f1-score'], reverse=True)

        return self.results

    def search_smooth_voting(self, test=None, results=None, decay=None, vote_thresh=None, var_thresh=None):
        """

        """
        assert results is not None or (test is not None and self.clf is not None and self.pipeline is not None), 'you MUST supply a test dataset or a list of results'
        assert decay is None or isinstance(decay, list), 'decay MUST be a list or None'
        assert vote_thresh is None or isinstance(vote_thresh, list), 'vote_thresh MUST be a list or None'
        assert var_thresh is None or isinstance(var_thresh, list), 'var_thresh MUST be a list or None'

        # compute base result to apply voting search to
        if results is None:
            results = [self.score(self.pipeline, test=test)]

        if decay is None:
            decay = [0.7, 0.75, 0.8]

        if vote_thresh is None:
            vote_thresh = [0.005, 0.01, 0.02, 0.05, 0.1]

        if var_thresh is None:
            var_thresh = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

        voting_results = []

        for result in results:
            for d in decay:
                for m in vote_thresh:
                    for r in var_thresh:
                        new_result = {
                            'pipeline': result['pipeline'],
                            'voting': {
                                'decay': d,
                                'mean_thresh': m,
                                'var_thresh': r
                            }
                        }
                        new_result.update(self._apply_generic_voting(
                            SmoothVoting(decay=d, vote_thresh=m, var_thresh=r),
                            result['y_true'],
                            result['y_pred']))

                        voting_results.append(new_result)

        self.results = sorted(voting_results, key=lambda res: res['f1-score'], reverse=True)
        return self.results

    def search_voting(self, test=None, results=None, short=None, long=None, quorum=None):
        """

        """
        assert results is not None or (test is not None and self.clf is not None and self.pipeline is not None), 'you MUST supply a test dataset or a list of results'
        assert short is None or isinstance(short, list), 'short MUST be a list or None'
        assert long is None or isinstance(long, list), 'long MUST be a list or None'
        assert quorum is None or isinstance(quorum, list), 'quorum MUST be a list or None'

        # compute base result to apply voting search to
        if results is None:
            results = [self.score(self.pipeline, test=test)]

        if short is None:
            short = [1, 5, 10, 15]

        if long is None:
            long = [1, 3, 5, 10]

        if quorum is None:
            quorum = [0.5, 0.7, 0.85]

        voting_results = []

        for result in results:
            for s in short:
                for l in long:
                    for q in quorum:
                        new_result = {
                            'pipeline': result['pipeline'],
                            'voting': {
                                'short': s,
                                'long': l,
                                'quorum': q
                            }
                        }
                        new_result.update(self._apply_generic_voting(
                            Voting(short=(s, s * q), long=(l, l * q)),
                            result['y_true'],
                            result['y_pred']))
                        voting_results.append(new_result)

        self.results = sorted(voting_results, key=lambda res: res['f1-score'], reverse=True)
        return self.results

    def score(self, pipeline, test):
        X_train, y_train = pipeline.X, pipeline.y
        X_test, y_test = pipeline.transform(test.X, test.y)

        y_pred = self.clf.fit(X_train, y_train).predict(X_test)

        return {
            'pipeline': pipeline,
            'accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1-score': f1_score(y_test, y_pred, average='weighted'),
            'y_true': y_test,
            'y_pred': y_pred
        }

    def search(self, train, test):
        """

        """
        assert isinstance(train, Dataset), 'train MUST be a dataset'
        assert isinstance(test, Dataset), 'test MUST be a dataset'

        pipelines = [
            [MinMaxScaler(), Window(length=8, shift=4), TSFRESH(num_features=train.num_features, k=5)],
            [MinMaxScaler(), Window(length=16, shift=4), TSFRESH(num_features=train.num_features, k=5)],
            [MinMaxScaler(), Window(length=24, shift=4), TSFRESH(num_features=train.num_features, k=5)],
            [MinMaxScaler(), Window(length=32, shift=4), TSFRESH(num_features=train.num_features, k=5)],
            [MinMaxScaler(), Window(length=8, shift=4), TSFRESH(num_features=train.num_features, k=10)],
            [MinMaxScaler(), Window(length=16, shift=4), TSFRESH(num_features=train.num_features, k=10)],
            [MinMaxScaler(), Window(length=24, shift=4), TSFRESH(num_features=train.num_features, k=10)],
            [MinMaxScaler(), Window(length=32, shift=4), TSFRESH(num_features=train.num_features, k=10)],
            [MinMaxScaler(), Window(length=8, shift=4), TSFRESH(num_features=train.num_features, k=15)],
            [MinMaxScaler(), Window(length=16, shift=4), TSFRESH(num_features=train.num_features, k=15)],
            [MinMaxScaler(), Window(length=24, shift=4), TSFRESH(num_features=train.num_features, k=15)],
            [MinMaxScaler(), Window(length=32, shift=4), TSFRESH(num_features=train.num_features, k=15)],
            [MinMaxScaler(), Window(length=8, shift=4), TSFRESH(num_features=train.num_features, k=0)],
            [MinMaxScaler(), Window(length=16, shift=4), TSFRESH(num_features=train.num_features, k=0)],
            [MinMaxScaler(), Window(length=24, shift=4), TSFRESH(num_features=train.num_features, k=0)],
            [MinMaxScaler(), Window(length=32, shift=4), TSFRESH(num_features=train.num_features, k=0)],
            [MinMaxScaler(), Window(length=8, shift=4), FFT(num_features=train.num_features)],
            [MinMaxScaler(), Window(length=16, shift=4), FFT(num_features=train.num_features)],
            [MinMaxScaler(), Window(length=32, shift=4), FFT(num_features=train.num_features)],
        ]

        classifiers = [
            RandomForestClassifier(n_estimators=5, max_depth=15, min_samples_leaf=15),
            RandomForestClassifier(n_estimators=10, max_depth=15, min_samples_leaf=15),
            RandomForestClassifier(n_estimators=15, max_depth=15, min_samples_leaf=15),
            RandomForestClassifier(n_estimators=5, max_depth=30, min_samples_leaf=15),
            RandomForestClassifier(n_estimators=10, max_depth=30, min_samples_leaf=15),
            RandomForestClassifier(n_estimators=15, max_depth=30, min_samples_leaf=15),
            XGBClassifier(n_estimators=5, max_depth=15),
            XGBClassifier(n_estimators=10, max_depth=15),
            XGBClassifier(n_estimators=15, max_depth=15),
            XGBClassifier(n_estimators=5, max_depth=30),
            XGBClassifier(n_estimators=10, max_depth=30),
            XGBClassifier(n_estimators=15, max_depth=30),
        ]

        # override defaults
        if self.clf is not None:
            classifiers = [self.clf]

        if self.pipeline is not None:
            pipelines = [self.pipeline.steps]

        n_combinations = len(pipelines) * len(classifiers)
        scores = []

        for i, steps in enumerate(pipelines):
            pipeline = Pipeline('Pipeline', train, steps=steps).fit()
            X_test, y_test = pipeline.transform(test.X, test.y)

            for j, clf in enumerate(classifiers):
                print('searching %d/%d...' % (i * len(classifiers) + j + 1, n_combinations))

                try:
                    y_pred = clf.clone().fit(pipeline.X, pipeline.y).predict(X_test)

                    for short_votes in self.short_votes:
                        for long_votes in self.long_votes:
                            for quorum in self.quorum:
                                scores.append({
                                    'pipeline': pipeline,
                                    'clf': clf,
                                    'voting': {
                                        'short': short_votes,
                                        'long': long_votes,
                                        'quorum': quorum
                                    },
                                    'scores': self._apply_voting(y_test, y_pred, short_votes, long_votes, short_votes * quorum, long_votes * quorum)
                                })
                except ValueError as err:
                    print('ValueError', err)

        return sorted(scores, key=lambda x: x['scores']['accuracy'], reverse=True)

    def _apply_voting(self, y_true, y_pred, s, l, S, L):
        voting = Voting(short=(s, S), long=(l, L))
        support = 0
        confidence = 0
        cf_true = []
        cf_pred = []
        time_to_prediction = 0
        times_to_prediction = []

        for yi_true, yi_pred in zip(y_true, y_pred):
            decision = voting.vote(yi_pred)
            time_to_prediction += 1

            if decision is None:
                continue

            times_to_prediction.append(time_to_prediction)
            time_to_prediction = 0
            support += 1
            cf_true.append(yi_true)
            cf_pred.append(decision)

            if decision == yi_true:
                confidence += 1

        times_to_prediction = np.asarray(times_to_prediction)

        return {
            'count': len(y_true),
            'support': support / len(y_true),
            'accuracy': 100 * confidence / support,
            'f1-score weightened': f1_score(cf_true, cf_pred, average='weighted'),
            'avg time to prediction': {
                'mean': times_to_prediction.mean(),
                'std': times_to_prediction.std()
            },
            'f1-score': f1_score(cf_true, cf_pred, average='weighted'),
            'latency': {
                'mean': times_to_prediction.mean(),
                'std': times_to_prediction.std()
            },
            'y_true': np.asarray(cf_true, dtype=int),
            'y_pred': np.asarray(cf_pred, dtype=int)
        }

    def _apply_generic_voting(self, voter, y_true, y_pred):
        support = 0
        confidence = 0
        cf_true = []
        cf_pred = []
        time_to_prediction = 0
        times_to_prediction = []

        for yi_true, yi_pred in zip(y_true, y_pred):
            decision = voter.vote(yi_pred)
            time_to_prediction += 1

            if decision is None:
                continue

            times_to_prediction.append(time_to_prediction)
            time_to_prediction = 0
            support += 1
            cf_true.append(yi_true)
            cf_pred.append(decision)

            if decision == yi_true:
                confidence += 1

        times_to_prediction = np.asarray(times_to_prediction)

        return {
            'count': len(y_true),
            'support': support / len(y_true),
            'accuracy': 100 * confidence / support,
            'f1-score weightened': f1_score(cf_true, cf_pred, average='weighted'),
            'avg time to prediction': {
                'mean': times_to_prediction.mean(),
                'std': times_to_prediction.std()
            },
            'f1-score': f1_score(cf_true, cf_pred, average='weighted'),
            'latency': {
                'mean': times_to_prediction.mean(),
                'std': times_to_prediction.std()
            },
            'y_true': np.asarray(cf_true, dtype=int),
            'y_pred': np.asarray(cf_pred, dtype=int)
        }