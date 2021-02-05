from os.path import exists

from time import sleep
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from eloquentarduino.ml.metrics.device import Runtime, Resources
from eloquentarduino.ml.metrics.device.BenchmarkPlotter import BenchmarkPlotter
from eloquentarduino.jupyter.project.Errors import BadBoardResponseError, ArduinoCliCommandError
from eloquentarduino.ml.data import CheckpointFile


class BenchmarkEndToEnd:
    """Run a moltitude of runtime benchmarks"""
    def __init__(self):
        """Init"""
        self.classifiers = []
        self.hidden_columns = []
        self.output_file = CheckpointFile(None, keys=['board', 'dataset', 'clf'])
        self.all_columns = [
            'board',
            'dataset',
            'clf',
            'n_features',
            'flash',
            'raw_flash',
            'flash_percent',
            'flash_score',
            'memory',
            'raw_memory',
            'memory_percent',
            'memory_score',
            'offline_accuracy',
            'online_accuracy',
            'inference_time'
        ]

    @property
    def result(self):
        """
        Return first result
        :return:
        """
        return None if self.df.empty else self.df.loc[0]

    @property
    def columns(self):
        """
        Get columns for DataFrame
        :return:
        """
        return [column for column in self.all_columns if column not in self.hidden_columns]

    @property
    def summary_columns(self):
        """
        Get important columns for DataFrame
        :return:
        """
        return [
            'board',
            'dataset',
            'clf',
            'flash',
            'memory',
            'offline_accuracy',
            'online_accuracy',
            'inference_time'
        ]

    @property
    def df(self):
        """
        Get results as pandas.DataFrame
        :return:
        """
        return pd.DataFrame(self.output_file.df, columns=self.columns)

    @property
    def sorted_df(self):
        """
        Get df sorted by board, dataset, classifier
        :return:
        """
        return self.df.sort_values(by=['board', 'dataset', 'clf'])

    @property
    def plot(self):
        """
        Get plotter utility
        :return:
        """
        return BenchmarkPlotter(self.df)

    def save_to(self, filename, overwrite=False):
        """
        Set file to save results
        :param filename:
        :param overwrite:
        """
        self.output_file = CheckpointFile(filename, keys=['board', 'dataset', 'clf'])
        if overwrite:
            self.output_file.clear()

    def set_precision(self, digits):
        """
        Set pandas precision
        :param digits:
        :return:
        """
        pd.set_option('precision', digits)

    def hide(self, *args):
        """
        Hide columns from DataFrame
        :param args:
        :return:
        """
        self.hidden_columns += args

    def benchmark(
            self,
            project,
            datasets,
            classifiers,
            boards=None,
            accuracy=True,
            runtime=False,
            offline_test_size=0.3,
            cross_val=3,
            online_test_size=20,
            repeat=5,
            port=None,
            upload_options={},
            random_state=0):
        """
        Run benchmark on the combinations of boards x datasets x classifiers
        :param project:
        :param boards:
        :param datasets:
        :param classifiers:
        :param accuracy:
        :param runtime:
        :param offline_test_size:
        :param cross_val:
        :param online_test_size:
        :param repeat:
        :param port:
        :param random_state:
        :return:
        """

        if boards is None:
            assert project.board.model is not None and len(project.board.model.fqbn) > 0, 'You MUST specify at least a board'
            boards = [project.board.model.fqbn]

        if port is None:
            # set 'auto' port if runtime is active
            if runtime and project.board.port is None:
                project.board.set_port('auto')
        else:
            project.board.set_port(port)

        n_run = 0
        n_combos = len(self.to_list(boards)) * len(self.to_list(datasets)) * len(self.to_list(classifiers))

        for board_name in self.to_list(boards):
            # set board
            project.board.set_model(board_name)
            board_name = project.board.name

            # if benchmarking runtime, we need the board to be connected
            # so be sure the sure has done the physical setup
            if runtime:
                input('Benchmarking board %s: press Enter to continue...' % board_name)

            # get the resources needed for the empty sketch
            baseline_resources = Resources(project).baseline()

            for dataset_name, (X, y) in self.to_list(datasets):
                for clf_name, clf in self.to_list(classifiers):
                    n_run += 1
                    project.logger.info('[%d/%d] Benchmarking %s x %s x %s', n_run, n_combos, board_name, dataset_name, clf_name)

                    if self.output_file.key_exists((board_name, dataset_name, clf_name)):
                        existing = self.output_file.get((board_name, dataset_name, clf_name))
                        # skip if we have all the data for the combination
                        if not runtime or float(existing.inference_time) > 0:
                            project.logger.debug('A checkpoint exists, skipping')
                            continue

                    # if clf is a lambda function, call with X, y arguments
                    if callable(clf):
                        clf_clone = clf(X, y)
                    else:
                        # make a copy of the original classifier
                        clf_clone = clone(clf)

                    # benchmark classifier accuracy (off-line)
                    if accuracy:
                        if cross_val:
                            cross_results = cross_validate(clf_clone, X, y, cv=cross_val, return_estimator=True)
                            offline_accuracy = cross_results['test_score'].mean()
                            # keep first classifier
                            clf_clone = cross_results['estimator'][0]
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=offline_test_size, random_state=random_state)
                            offline_accuracy = clf_clone.fit(X_train, y_train).score(X_test, y_test)
                    else:
                        offline_accuracy = 0
                        clf_clone.fit(X, y)

                    try:
                        resources_benchmark = Resources(project).benchmark(clf_clone, x=X[0])
                    except NotFittedError:
                        project.logger.error('Classifier not fitted, cannot benchmark')
                        continue
                    except ArduinoCliCommandError:
                        project.logger.error('Arduino CLI reported an error')
                        continue
                    except Exception as err:
                        project.logger.error('Generic error', err)
                        continue

                    # benchmark on-line inference time and accuracy
                    if runtime:
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=online_test_size, random_state=random_state)
                            runtime_benchmark = Runtime(project).benchmark(clf_clone, X_test, y_test, repeat=repeat, upload_options=upload_options)
                            project.logger.info('Benchmarked runtime inference')
                        except BadBoardResponseError as e:
                            project.logger.error(e)
                            runtime_benchmark = Runtime.empty()
                    else:
                        runtime_benchmark = Runtime.empty()

                    self.classifiers.append(clf_clone)
                    self.add_result(
                        board=board_name,
                        dataset=dataset_name,
                        clf=clf_name,
                        shape=X.shape,
                        offline_accuracy=offline_accuracy,
                        resources=resources_benchmark,
                        runtime=runtime_benchmark,
                        baseline=baseline_resources)

                    sleep(2)

        return self

    def add_result(
            self,
            board,
            dataset,
            clf,
            shape,
            offline_accuracy,
            resources,
            runtime,
            baseline
    ):
        """
        Add result to list
        :param board:
        :param dataset:
        :param clf:
        :param shape:
        :param offline_accuracy:
        :param resources:
        :param runtime:
        :param baseline:
        :return:
        """
        raw_flash = resources['flash']
        raw_memory = resources['memory']

        if baseline:
            resources['flash'] -= baseline['flash']
            resources['memory'] -= baseline['memory']

        result = {
            'board': board,
            'dataset': dataset,
            'clf': clf,
            'n_features': shape[1],
            'flash': resources['flash'],
            'memory': resources['memory'],
            'raw_flash': raw_flash,
            'raw_memory': raw_memory,
            'flash_percent': resources['flash_percent'],
            'memory_percent': resources['memory_percent'],
            'flash_score': offline_accuracy * (1 - resources['flash_percent']),
            'memory_score': offline_accuracy * (1 - resources['memory_percent']),
            'offline_accuracy': offline_accuracy,
            'online_accuracy': runtime['online_accuracy'],
            'inference_time': runtime['inference_time']
        }
        self.output_file.set((board, dataset, clf), result)

    def to_list(self, x):
        """
        Convert argument to list, if not already
        :param x:
        :return:
        """
        return x if isinstance(x, list) else [x]