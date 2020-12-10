from inspect import getmro
from itertools import product
from sklearn.base import clone
from collections import namedtuple
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd
from eloquentarduino.ml.metrics.device import Resources, Runtime
from eloquentarduino.ml.data import CheckpointFile


class GridSearch:
    """
    Perform grid search on supported classifiers
    """
    DEFAULT_HYPERPARAMETERS = {
        'DecisionTreeClassifier': [{
            'max_depth': [5, 10, 20, None],
            'min_samples_leaf': [1, 5, 10],
            'max_features': [0.5, 0.75, "sqrt", None]
        }],
        'RandomForestClassifier': [{
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_leaf': [1, 5, 10],
            'max_features': [0.5, 0.75, "sqrt", None]
        }],
        'XGBClassifier': [{
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'eta': [0.1, 0.3, 0.7],
            'gamma': [0, 1, 10]
        }],
        'GaussianNB': [{
            'var_smoothing': [1e-5, 1e-7, 1e-9]
        }],
        'LogisticRegression': [{
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1],
            'max_iter': [1e3, 1e4, 1e5]
        }],
        'SVC': [{
            'kernel': ['linear'],
            'C': [0.01, 0.1, 1],
            'max_iter': [1000, -1]
        }, {
            'kernel': ['poly'],
            'degree': [2, 3],
            'gamma': [0.001, 0.1, 1, 'auto'],
            'C': [0.01, 0.1, 1],
            'max_iter': [1000, -1]
        }, {
            'kernel': ['rbf'],
            'gamma': [0.001, 0.1, 1, 'auto'],
            'C': [0.01, 0.1, 1],
            'max_iter': [1000, -1]
        }]
    }

    def __init__(self, dataset):
        """
        :param dataset:
        """
        assert isinstance(dataset, tuple), 'Dataset MUST be a tuple, either (name, X, y) or (X, y)'
        assert len(dataset) == 2 or len(dataset) == 3, 'Dataset MUST be a tuple, either (name, X, y) or (X, y)'
        self.dataset = dataset if len(dataset) == 3 else ('Unknown dataset', dataset[0], dataset[1])
        self.classifiers = []
        self.constraints = {
            'offline': [],
            'resources': [],
            'runtime': []
        }
        self.features = set([])
        self.output_file = CheckpointFile(None, keys=['board', 'dataset', 'clf_description'])
        self.candidates = []
        self.runs = 0
        self.Result = namedtuple('GridSearchResult', 'clf clf_description accuracy min_accuracy max_accuracy flash flash_percent memory memory_percent inference_time passes')

    @property
    def df(self):
        """
        Get candidates as pandas.DataFrame
        :return: pandas.DataFrame
        """
        if self.output_file.filename is not None:
            return self.output_file.df

        df = pd.DataFrame(self.candidates)
        if 'clf' in df.columns:
            df = df.drop(columns=['clf'])
        return df

    @property
    def df_passes(self):
        """
        Get candidates that pass the constraints
        """
        return self.df[self.df.passes]

    def add_classifier(self, clf, only=None, merge=None):
        """
        Add classifier to list of candidates
        :param only: search ONLY these params
        :param merge: search ALSO these params, plus the defaults
        """
        if only is not None:
            search_params = only if isinstance(only, list) else [only]
        else:
            search_params = None
            defaults = GridSearch.DEFAULT_HYPERPARAMETERS

            # merge defaults with user supplied
            for clf_type, params in defaults.items():
                if self._check_type(clf, clf_type):
                    search_params = params
                    if merge is not None:
                        search_params += merge if isinstance(merge, list) else [merge]
                    break

            assert search_params is not None, 'Cannot find default search params for %s, you MUST set only=' % type(clf)

        self.classifiers.append((clf, search_params))

    def save_to(self, filename, overwrite=False):
        """
        Set file to save results
        :param filename:
        :param overwrite:
        """
        self.output_file = CheckpointFile(filename, keys=['board', 'dataset', 'clf_description'])
        if overwrite:
            self.output_file.clear()

    def min_accuracy(self, acc):
        """
        Add constraint on min accuracy
        :param acc:
        """
        self.add_offline_constraint(lambda result: result.accuracy >= acc)

    def max_flash(self, flash):
        """
        Add constraint on flash size
        :param flash:
        """
        self.add_resources_constraint(lambda result: result.flash <= flash)

    def max_flash_percent(self, percent):
        """
        Add constraint on flash size
        :param percent:
        """
        self.add_resources_constraint(lambda result: result.flash_percent <= percent)

    def max_memory(self, memory):
        """
        Add constraint on memory size
        :param memory:
        """
        self.add_resources_constraint(lambda result: result.memory <= memory)

    def max_memory_percent(self, percent):
        """
        Add constraint on memory size
        :param percent:
        """
        self.add_resources_constraint(lambda result: result.memory_percent <= percent)

    def max_inference_time(self, micros):
        """
        Add constraint on inference time
        :param micros:
        """
        self.add_runtime_constraint(lambda result: result.inference_time <= micros)

    def add_offline_constraint(self, constraint):
        """
        Add constraint to offline result
        :param constraint:
        """
        self._add_constraint('offline', constraint)

    def add_resources_constraint(self, constraint):
        """
        Add constraint to resources result
        :param constraint:
        """
        self._add_constraint('resources', constraint)

    def add_runtime_constraint(self, constraint):
        """
        Add constraint to runtime result
        :param constraint:
        """
        self._add_constraint('runtime', constraint)

    def search(self, project, cv=3):
        """
        Perform search
        :param project:
        :param cv: cross validation splits
        """
        dataset_name, X, y = self.dataset
        board_name = project.board.fqbn
        
        for base_clf, search_bags in self.classifiers:
            project.logger.debug('Tuning %s', type(base_clf).__name__)
            # naive implementation of grid search
            for search_params in search_bags:
                for combo in product(*search_params.values()):
                    current_params = {k: v for k, v in zip(search_params.keys(), combo)}
                    params_string = ', '.join(['%s=%s' % (k, str(v)) for k, v in current_params.items()])
                    clf_description = '%s (%s)' % (type(base_clf).__name__, params_string)
                    project.logger.debug('Benchmarking %s', clf_description)

                    if self.output_file.key_exists((board_name, dataset_name, clf_description)):
                        project.logger.debug('A checkpoint exists, skipping')
                        continue

                    self.runs += 1

                    clf = clone(base_clf)
                    clf.set_params(**current_params)
                    crossval = cross_validate(estimator=clf, X=X, y=y, cv=cv, return_estimator=True)
                    best_idx = np.argmax(crossval['test_score'])
                    result = self.Result(
                        clf=crossval['estimator'][best_idx],
                        clf_description=clf_description,
                        accuracy=crossval['test_score'].mean(),
                        min_accuracy=crossval['test_score'].min(),
                        max_accuracy=crossval['test_score'].max(),
                        flash=0,
                        flash_percent=0,
                        memory=0,
                        memory_percent=0,
                        inference_time=0,
                        passes=True)

                    passes = True

                    # apply offline constraints
                    for constraint in self.constraints['offline']:
                        if not constraint(result):
                            project.logger.debug('%s didn\'t passed the offline constraints', clf_description)
                            passes = False
                            break
                    if not passes:
                        self._checkpoint(board_name, dataset_name, clf_description, result, False)
                        continue

                    # apply resources constraints
                    if len(self.constraints['resources']):
                        resources = Resources(project).benchmark(result.clf, x=X[0])
                        result = result._replace(
                            flash=resources['flash'],
                            flash_percent=resources['flash_percent'],
                            memory=resources['memory'],
                            memory_percent=resources['memory_percent']
                        )
                        for constraint in self.constraints['resources']:
                            if not constraint(result):
                                project.logger.debug('%s didn\'t passed the resources constraints', clf_description)
                                passes = False
                                break
                    if not passes:
                        self._checkpoint(board_name, dataset_name, clf_description, result, False)
                        continue

                    # apply runtime constraints
                    if len(self.constraints['runtime']):
                        runtime = Runtime(project).benchmark(result.clf, X[:3], y[:3], repeat=10, compile=False)
                        result = result._replace(
                            inference_time=runtime['inference_time']
                        )
                        for constraint in self.constraints['runtime']:
                            if not constraint(result):
                                project.logger.debug('%s didn\'t passed the resources constraints', clf_description)
                                passes = False
                                break
                    if not passes:
                        self._checkpoint(board_name, dataset_name, clf_description, result, False)
                        continue

                    # all constraints passed, add to candidates
                    project.logger.debug('%s passed all constraints, added to the list of candidates', clf_description)
                    self._checkpoint(board_name, dataset_name, clf_description, result, True)

    def _check_type(self, clf, *classes):
        """
        Check if clf is instance of given class
        :return: bool
        """
        for klass in classes:
            if type(clf).__name__ == klass:
                return True
            for T in getmro(type(clf)):
                if T.__name__ == klass:
                    return True
        return False

    def _add_constraint(self, env, constraint):
        """
        Add constraint to results to be considered
        :param env: when the constraint should run (offline, resources, runtime)
        :param constraint:
        """
        assert callable(constraint), 'constraint MUST be a function'
        self.constraints[env].append(constraint)

    def _checkpoint(self, board_name, dataset_name, clf_name, result, passes):
        """
        Save checkpoint for search
        :param board_name:
        :param dataset_name:
        :param clf_name:
        :param result:
        :param passes:
        """
        result = result._replace(passes=passes)
        result = {**result._asdict(), **{
            'board': board_name,
            'dataset': dataset_name
        }}
        del result['clf']
        self.output_file.set((board_name, dataset_name, clf_name), result)
        self.candidates.append(result)
