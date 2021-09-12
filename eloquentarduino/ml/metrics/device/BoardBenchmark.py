import re
import numpy as np

from micromlgen import port as port_clf
from sklearn.model_selection import cross_validate

from eloquentarduino.jupyter import Project
from eloquentarduino.jupyter.project import BoardConfiguration
from eloquentarduino.jupyter.project.Errors import BadBoardResponseError, BoardBenchmarkAlreadyExists
from eloquentarduino.ml.data import CheckpointFile
from eloquentarduino.ml.data import Dataset
from eloquentarduino.ml.classification.abstract.Classifier import Classifier
from eloquentarduino.utils import jinja


class BoardBenchmark:
    _cache = {}

    """
    Benchmark classifier + dataset on a given board
    """
    def __init__(self, board, dataset, classifier):
        """
        :param board: BoardConfiguration
        :param dataset: Dataset
        :param classifier: Classifier
        """
        assert isinstance(board, BoardConfiguration), 'board MUST be a jupyter.project.BoardConfiguration instance'
        assert isinstance(dataset, Dataset), 'dataset MUST be a ml.data.Dataset instance'
        assert isinstance(classifier, Classifier), 'classifier MUST be a ml.Classifier instance'

        self.board = board
        self.dataset = dataset
        self.classifier = classifier
        self.key = (self.board.name, self.dataset.name, self.classifier.name)

    def benchmark(
            self,
            port=None,
            project=None,
            inference_time=False,
            save_to=None,
            exists_ok=True,
            exists_overwrite=False,
            cv=3,
            before_upload=None,
            after_upload=None):
        """

        """
        if inference_time:
            assert port is not None or (project is not None and project.board is not None and project.board.port is not None), 'You MUST set a port'

        save_to = CheckpointFile(save_to, keys=['board', 'dataset', 'clf'])

        if save_to.key_exists(self.key) and exists_ok:
            return

        if save_to.key_exists(self.key) and not exists_overwrite:
            raise BoardBenchmarkAlreadyExists(self.key)

        if project is None:
            project = Project()

        if inference_time and port:
            project.board.set_port(port)

        # benchmark offline accuracy
        X = self.dataset.X
        y = self.dataset.y
        idx = np.arange(len(X))[::(len(X) // 5)][:5]
        X_test = X[idx]
        y_test = y[idx]
        cross_results = cross_validate(self.classifier.generator(X, y), X, y, cv=cv, return_estimator=True)
        offline_accuracy = cross_results['test_score'].mean()
        clf = cross_results['estimator'][0]

        benchmark = {
            'board': self.board.name,
            'dataset': self.dataset.name,
            'clf': self.classifier.name,
            'fqbn': '',
            'cpu_speed': self.board.cpu_speed,
            'cpu_family': self.board.cpu_family,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'offline_accuracy': offline_accuracy,
            'inference_time': 0
        }

        with project.tmp_project() as tmp:
            tmp.board.set_model(self.board)

            benchmark['fqbn'] = tmp.board.fqbn
            cache_key = (self.board.name, self.dataset.name)

            if cache_key not in BoardBenchmark._cache:
                BoardBenchmark._cache[cache_key] = self.get_baseline(tmp, X_test)

            baseline = BoardBenchmark._cache.get(cache_key)

            sketch = jinja('metrics/Resources.jinja', {'X': X_test})
            ported = port_clf(clf, classname='Classifier')
            tmp.files.add('%s.ino' % tmp.name, contents=sketch, exists_ok=True)
            tmp.files.add('Classifier.h', contents=ported, exists_ok=True)

            resources = self._parse_resources(tmp)
            resources['flash_increment'] = resources['flash'] - baseline['flash']
            resources['memory_increment'] = resources['memory'] - baseline['memory']
            resources['flash_increment_percent'] = float(resources['flash_increment']) / resources['flash_max'] if resources['flash_max'] > 0 else 0
            resources['memory_increment_percent'] = float(resources['memory_increment']) / resources['memory_max'] if resources['memory_max'] > 0 else 0
            benchmark.update(resources)

            if inference_time:
                sketch = jinja('metrics/Runtime.jinja', {'X': X_test, 'y': y_test})
                ported = port_clf(clf, classname='Classifier')

                tmp.files.add(tmp.ino_name, contents=sketch, exists_ok=True)
                tmp.files.add('Classifier.h', contents=ported, exists_ok=True)
                if callable(before_upload):
                    before_upload()
                tmp.upload(success_message='')
                if callable(after_upload):
                    after_upload(tmp)
                benchmark.update(self._parse_inference_time(tmp))

        save_to.set(self.key, benchmark)

        return benchmark

    def get_baseline(self, project, X):
        """
        Get resources for an empty sketch
        :param project:
        :param X:
        :return: dict of resources
        """
        project.logger.debug('benchmarking empty sketch to get a baseline')
        with project.tmp_project() as tmp:
            tmp.files.add('%s.ino' % tmp.name, contents=jinja('metrics/Baseline.jinja', {'X': X}), exists_ok=True)
            return self._parse_resources(tmp)

    def _parse_resources(self, project):
        """
        Actually benchmark the current sketch
        :param project:
        :return: dict {flash, flash_max, flash_percent, memory, memory_max, memory_percent}
        """
        compile_log = project.compile()
        flash_pattern = r'Sketch uses (\d+) bytes.+?Maximum is (\d+)'
        memory_pattern = r'Global variables use (\d+).+?Maximum is (\d+)'
        flash_match = re.search(flash_pattern, compile_log.replace("\n", ""))
        memory_match = re.search(memory_pattern, compile_log.replace("\n", ""))

        if flash_match is None and memory_match is None:
            raise RuntimeError('Cannot parse compilation log: %s' % compile_log)

        flash, flash_max = [int(g) for g in flash_match.groups()] if flash_match is not None else [0, 0]
        memory, memory_max = [int(g) for g in memory_match.groups()] if memory_match is not None else [0, 0]

        return {
            'flash': flash,
            'flash_max': flash_max,
            'flash_percent': float(flash) / flash_max if flash_max > 0 else 0,
            'memory': memory,
            'memory_max': memory_max,
            'memory_percent': float(memory) / memory_max if memory_max > 0 else 0,
        }

    def _parse_inference_time(self, project):
        """
        Parse response from board
        :param project:
        :return: dict {inference_time}
        """
        for i in range(0, 3):
            response = project.serial.read_until('======', timeout=5)
            match = re.search(r'inference time = ([0-9.]+) micros[\s\S]', response)

            if match is not None:
                return {
                    'inference_time': float(match.group(1)),
                }

        raise BadBoardResponseError('Unexpected response during runtime inference time benchmark')
