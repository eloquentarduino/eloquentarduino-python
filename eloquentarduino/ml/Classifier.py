import numpy as np
import eloquentarduino
import micromlgen
from eloquentarduino.utils import jinja
from eloquentarduino.ml.metrics.device.parsers import CompileLogParser


class Classifier:
    _baseline = None

    """
    Abstraction of a classifier
    """
    def __init__(self, name, generator):
        """
        :param name:
        :param generator: function that returns a classifier. Must accept X, y
        """
        self.name = name
        self.generator = generator if callable(generator) else lambda X, y: generator
        self.clf = self.dummy_clf

    @property
    def dummy_clf(self):
        """
        Create a dummy instance of the classifier
        """
        return self.generator(np.zeros((10, 10), dtype=np.float), np.zeros(10, dtype=np.int))

    def is_tf(self):
        """
        Test if wrapped classifier is a Tf model
        """
        return type(self.dummy_clf).__name__ == 'TfMicro'

    def fit(self, X, y, **kwargs):
        """
        Fit classifier
        :param X:
        :param y:
        """
        self.clf = self.generator(X, y)
        print('clf', self.clf)
        self.clf.fit(X, y, **kwargs)

    def port(self, **kwargs):
        """
        Port classifier to C++
        """
        return self.clf.port(**kwargs) if self.is_tf() else micromlgen.port(self.clf, **kwargs)

    def benchmark_resources(self, X, project=eloquentarduino.project):
        """
        Compute resources needed to compile a sketch that uses this classifier
        :param X:
        :param project:
        """
        with project.tmp_project() as tmp:
            Classifier.benchmark_baseline(tmp, X)

            # compile a benchmarking sketch and get the resources needed
            template_folder = 'tf' if self.is_tf() else 'sklearn'
            sketch = jinja('benchmarks/%s/Resources.jinja' % template_folder, {'X': X})
            ported = self.port(classname='Classifier')
            tmp.files.add('%s.ino' % tmp.name, contents=sketch, exists_ok=True)
            tmp.files.add('Classifier.h', contents=ported, exists_ok=True)

            return CompileLogParser(project=tmp).sub(Classifier._baseline).info

    @classmethod
    def benchmark_baseline(cls, project, X):
        """
        Create an empty sketch to get the bare minimum resources needed
        """
        if cls._baseline is None:
            with project.tmp_project() as tmp:
                tmp.files.add('%s.ino' % tmp.name, contents=jinja('metrics/Baseline.jinja', {'X': X}), exists_ok=True)
                cls._baseline = CompileLogParser(project=tmp).info



