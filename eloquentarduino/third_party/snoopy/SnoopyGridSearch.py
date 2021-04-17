import numpy as np
from sklearn.metrics import f1_score
from eloquentarduino.ml.data import Dataset
from eloquentarduino.ml.data.preprocessing.pipeline import *
from eloquentarduino.ml.classification.sklearn import *
from eloquentarduino.third_party.snoopy.Voting import Voting


class SnoopyGridSearch:
    """
    Perform grid search on well-known configurations
    """
    def __init__(self, clf=None, pipeline=None):
        self.clf = clf
        self.pipeline = pipeline

    def search(self, train, test):
        """

        """
        assert isinstance(train, Dataset), 'train MUST be a dataset'
        assert isinstance(test, Dataset), 'test MUST be a dataset'

        pipelines = [
            #[Window(length=8, shift=4), TSFRESH(num_features=train.num_features, k=5)],
            #[Window(length=16, shift=4), TSFRESH(num_features=train.num_features, k=5)],
            #[Window(length=24, shift=4), TSFRESH(num_features=train.num_features, k=5)],
            #[Window(length=32, shift=4), TSFRESH(num_features=train.num_features, k=5)],
            #[Window(length=8, shift=4), TSFRESH(num_features=train.num_features, k=10)],
            #[Window(length=16, shift=4), TSFRESH(num_features=train.num_features, k=10)],
            #[Window(length=24, shift=4), TSFRESH(num_features=train.num_features, k=10)],
            #[Window(length=32, shift=4), TSFRESH(num_features=train.num_features, k=10)],
            #[Window(length=8, shift=4), TSFRESH(num_features=train.num_features, k=15)],
            #[Window(length=16, shift=4), TSFRESH(num_features=train.num_features, k=15)],
            #[Window(length=24, shift=4), TSFRESH(num_features=train.num_features, k=15)],
            #[Window(length=32, shift=4), TSFRESH(num_features=train.num_features, k=15)],
            [Window(length=8, shift=4), TSFRESH(num_features=train.num_features, k=0)],
            [Window(length=16, shift=4), TSFRESH(num_features=train.num_features, k=0)],
            [Window(length=24, shift=4), TSFRESH(num_features=train.num_features, k=0)],
            [Window(length=32, shift=4), TSFRESH(num_features=train.num_features, k=0)],
            [Window(length=8, shift=4), FFT(num_features=train.num_features)],
            [Window(length=16, shift=4), FFT(num_features=train.num_features)],
            [Window(length=32, shift=4), FFT(num_features=train.num_features)],
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

                    for short_votes in [1, 5, 10, 15]:
                        for long_votes in [1, 3, 5, 10]:
                            for quorum in [0.5, 0.7, 0.85]:
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
            }
        }
