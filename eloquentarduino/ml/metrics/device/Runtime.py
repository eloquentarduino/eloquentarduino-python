import re
import numpy as np
import pandas as pd
from micromlgen import port
from time import sleep

from eloquentarduino.ml.metrics.plot import Barplot
from eloquentarduino.utils import jinja
from eloquentarduino.jupyter.project.Errors import BadBoardResponseError


class Runtime:
    """
    Compute on-device inference time and accuracy
    """
    def __init__(self, project):
        """
        :param project:
        """
        self.project = project

    @property
    def df(self):
        """
        Convert results to DataFrame
        :return: pd.DataFrame
        """
        data = [{
            'dataset': r['dataset'],
            'clf': r['clf'],
            'score': r['score'],
            'time': r['time']
        } for r in self.results]

        return pd.DataFrame(data, columns=['dataset', 'clf', 'score', 'time'])

    @staticmethod
    def empty():
        """
        Get empty response
        :return:
        """
        return {
            'inference_time': 0,
            'online_accuracy': 0
        }

    def benchmark(self, clf, X_test=None, y_test=None, n_features=1, n_samples=20, repeat=1, upload_options={}):
        """
        Benchmark on-line inference time for a classifier
        :param clf:
        :param X_test:
        :param y_test:
        :param n_features:
        :param n_samples:
        :param repeat:
        :param compile:
        :param upload_options:
        :return:
        """
        if X_test is None or y_test is None:
            assert n_features > 0, 'n_features MUST be positive when X_test is not set'
            assert n_samples > 0, 'n_samples MUST be positive when X_test is not set'
            X_test = np.random.random((n_samples, n_features))
            y_test = np.random.random_integers(0, 1, n_samples)

        with self.project.tmp_project() as tmp:
            # upload benchmarking sketch
            sketch = jinja('metrics/Runtime.jinja', {
                'X_test': X_test,
                'y_test': y_test,
                'repeat': repeat
            })
            ported = port(clf, classname='Classifier')

            tmp.files.add(tmp.ino_name, contents=sketch, exists_ok=True)
            tmp.files.add('Classifier.h', contents=ported, exists_ok=True)
            tmp.upload(**upload_options)

            # parse serial output
            # since we can miss the first response, try a few times
            for i in range(0, 3):
                response = tmp.serial.read_until('======', timeout=8)
                match = re.search(r'inference time = ([0-9.]+) micros[\s\S]+?Score = ([0-9.]+)', response)

                if match is not None:
                    return {
                        'inference_time': float(match.group(1)),
                        'online_accuracy': float(match.group(2))
                    }

        self.project.logger.error('Failed to parse response: %s' % response)
        raise BadBoardResponseError('Unexpected response during runtime inference time benchmark')

    def plot(self, metric, sort=True, scale='linear', **kwargs):
        """
        Plot given metric
        :param scale:
        :param metric:
        :param sort:
        :return:
        """
        assert metric in self.df.columns and metric != 'clf', 'metric MUST be one of the columns'

        Barplot(self.df, x='clf', y=metric, sort=sort, scale=scale, **kwargs)