from os.path import exists

import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.base import clone

from eloquentarduino.ml.metrics.device import Runtime, Resources
from eloquentarduino.ml.metrics.device.BenchmarkPlotter import BenchmarkPlotter


class BenchmarkEndToEnd:
    """Run a moltitude of runtime benchmarks"""
    def __init__(self):
        """Init"""
        self.results = []
        self.classifiers = []
        self.hidden_columns = []

    @property
    def result(self):
        """
        Return first result
        :return:
        """
        return self.results[0] if len(self.results) else None

    @property
    def columns(self):
        """
        Get columns for DataFrame
        :return:
        """
        columns = [
            'board',
            'dataset',
            'clf',
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
        # hide columns
        columns = [column for column in columns if column not in self.hidden_columns]
        # return all columns if not run yet
        if len(self.results) == 0:
            return columns
        # return only columns that got computed
        return [column for column in columns if column in self.results[0]]

    @property
    def summary_columns(self):
        """
        Get important columns for DataFrame
        :return:
        """
        columns = [
            'board',
            'dataset',
            'clf',
            'flash',
            'memory',
            'offline_accuracy',
            'online_accuracy',
            'inference_time'
        ]
        return [column for column in columns if column in self.columns]

    @property
    def df(self):
        """
        Get results as pandas.DataFrame
        :return:
        """
        return pd.DataFrame(self.results, columns=self.columns)

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

    def load(self, checkpoint_file):
        """
        Load results from files
        :param checkpoint_file:
        :return: pd.DataFrame
        """
        assert exists(checkpoint_file), 'file %s NOT FOUND' % checkpoint_file
        df = pd.read_csv(checkpoint_file)
        self.results = df.to_dict('records')
        return df

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
            boards,
            datasets,
            classifiers,
            accuracy=True,
            runtime=False,
            offline_test_size=0.3,
            cross_val=3,
            online_test_size=20,
            repeat=5,
            checkpoint_file=None,
            save_checkpoints=True,
            port=None,
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
        :param online_test_size:
        :param repeat:
        :param checkpoint_file:
        :param save_checkpoints:
        :param port:
        :param random_state:
        :return:
        """
        checkpoints = None

        if checkpoint_file is not None and exists(checkpoint_file):
            checkpoints = self.load(checkpoint_file)

        for board_name in self.to_list(boards):
            # if benchmarking runtime, we need the board to be connected
            # so be sure the sure has done the physical setup
            if runtime:
                input('Benchmarking board %s: press Enter to continue...' % board_name)

            # set board
            project.board.set_model(board_name)
            board_name = project.board.model.name

            # get the resources needed for the empty sketch
            baseline_resources = Resources(project).baseline()

            for dataset_name, (X, y) in self.to_list(datasets):
                for clf_name, clf in self.to_list(classifiers):
                    project.log('Benchmarking %s x %s x %s' % (board_name, dataset_name, clf_name))

                    # make a copy of the original classifier
                    clf = clone(clf)

                    # if a checkpoint exists, skip benchmarking
                    if self.checkpoint_exists(checkpoints, board=board_name, dataset=dataset_name, clf=clf_name, runtime=runtime):
                        project.log('A checkpoint exists, skipping benchmarking')
                        continue

                    # benchmark classifier accuracy (off-line)
                    if accuracy:
                        if cross_val:
                            cross_results = cross_validate(clf, X, y, cv=cross_val, return_estimator=True)
                            offline_accuracy = cross_results['test_score'].mean()
                            # keep first classifier
                            clf = cross_results['estimator'][0]
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=offline_test_size, random_state=random_state)
                            offline_accuracy = clf.fit(X_train, y_train).score(X_test, y_test)
                    else:
                        offline_accuracy = 0
                        clf.fit(X, y)

                    project.board.set_port(port if port is not None else ('auto' if runtime else '/dev/ttyUSB99'))
                    resources_benchmark = Resources(project).benchmark(clf, x=X[0])

                    # benchmark on-line inference time and accuracy
                    if runtime:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=online_test_size, random_state=random_state)
                        runtime_benchmark = Runtime(project).benchmark(clf, X_test, y_test, repeat=repeat, compile=False)
                    else:
                        runtime_benchmark = Runtime.empty()

                    self.classifiers.append(clf)
                    self.add_result(
                        board=board_name,
                        dataset=dataset_name,
                        clf=clf_name,
                        shape=X.shape,
                        offline_accuracy=offline_accuracy,
                        resources=resources_benchmark,
                        runtime=runtime_benchmark,
                        baseline=baseline_resources,
                        checkpoints=checkpoints,
                        checkpoint_file=(checkpoint_file if save_checkpoints else None))

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
            baseline,
            checkpoints,
            checkpoint_file
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
        :param checkpoints
        :param checkpoint_file
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

        self.results.append(result)

        # save checkpoint
        if checkpoint_file is not None:
            self.df.to_csv(checkpoint_file, index=False)

    def to_list(self, x):
        """
        Convert argument to list, if not already
        :param x:
        :return:
        """
        return x if isinstance(x, list) else [x]

    def checkpoint_exists(self, checkpoints, board, dataset, clf, runtime):
        """
        Check if a checkpoint for the given combo exists
        :param checkpoints:
        :param board:
        :param dataset:
        :param clf:
        :param runtime:
        :return:
        """
        if checkpoints is None:
            return False
        match = (checkpoints['board'] == board) & (checkpoints['dataset'] == dataset) & (checkpoints['clf'] == clf)
        checkpoint = checkpoints.loc[match]

        if checkpoint.empty:
            return False

        # if benchmarking runtime but no runtime is available, drop the row and recalculate
        if runtime and checkpoint.iloc[0]['inference_time'] == 0:
            self.results.pop(checkpoint.index[0])
            return False

        return True
