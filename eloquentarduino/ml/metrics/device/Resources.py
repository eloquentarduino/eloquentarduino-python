import re
import numpy as np
from micromlgen import port

from eloquentarduino.ml.metrics.plot import Barplot
from eloquentarduino.utils import jinja


class Resources:
    """
    Compute on-device metrics
    """
    def __init__(self, project):
        """
        :param project:
        """
        self.project = project

    def baseline(self):
        """
        Create an empty sketch to get the bare minimum resources needed
        :return:
        """
        with self.project.tmp_project() as tmp:
            tmp.files.add('%s.ino' % tmp.name, contents=jinja('metrics/Baseline.jinja'), exists_ok=True)
            return self._benchmark_current(tmp)

    def benchmark(self, clf, x=None, n_features=1):
        """
        Run the benchmark for a given classifier
        :param clf:
        :param x:
        :param n_features:
        :return:
        """
        if x is None:
            x = np.random.random(n_features)

        with self.project.tmp_project() as tmp:
            sketch = jinja('metrics/Resources.jinja', {'x': x})
            ported = port(clf, classname='Classifier')
            tmp.files.add('%s.ino' % tmp.name, contents=sketch, exists_ok=True)
            tmp.files.add('Classifier.h', contents=ported, exists_ok=True)
            return self._benchmark_current(tmp)

    def _benchmark_current(self, tmp):
        """
        Actually benchmark the current sketch
        :return: dict {flash, flash_max, flash_percent, memory, memory_max, memory_percent}
        """
        compile_log = tmp.compile()
        flash_pattern = r'Sketch uses (\d+) bytes.+?Maximum is (\d+)'
        memory_pattern = r'Global variables use (\d+).+?Maximum is (\d+)'
        flash_match = re.search(flash_pattern, compile_log.replace("\n", ""))
        memory_match = re.search(memory_pattern, compile_log.replace("\n", ""))

        if flash_match is None and memory_match is None:
            raise RuntimeError('Cannot parse compilation log: %s' % compile_log)

        flash, flash_max = [int(g) for g in flash_match.groups()] if flash_match is not None else [0, 1]
        memory, memory_max = [int(g) for g in memory_match.groups()] if memory_match is not None else [0, 1]

        return {
            'flash': flash,
            'flash_max': flash_max,
            'flash_percent': float(flash) / flash_max,
            'memory': memory,
            'memory_max': memory_max,
            'memory_percent': float(memory) / memory_max,
        }

    def plot(self, metric, sort=True, **kwargs):
        """
        Plot given metric
        :param metric:
        :param sort:
        :return:
        """
        assert metric in self.df.columns and metric != 'clf', 'metric MUST be one of the columns'

        Barplot(self.df, x='clf', y=metric, sort=sort, **kwargs)