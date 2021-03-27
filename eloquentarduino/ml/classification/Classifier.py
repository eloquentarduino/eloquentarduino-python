import numpy as np
import eloquentarduino
import micromlgen
from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split
from keras.utils import to_categorical
from eloquentarduino.utils import jinja
from eloquentarduino.ml.data import Dataset
from eloquentarduino.ml.metrics.device.parsers import CompileLogParser


class Classifier:
    """
    Abstraction of a classifier
    @deprecated
    """
    def __init__(self, name, generator):
        """
        :param name:
        :param generator: function that returns a classifier (must accept X and y)
        """
        assert callable(generator) or generator.__class__.__module__.startswith('sklearn.'), 'generator MUST be a function or a sklearn classifier: got %s' % generator.__class__

        self.name = name
        self.generator = generator if callable(generator) else lambda *args, **kwargs: clone(generator)
        self.clf = None

    def is_tf(self):
        """
        Test if wrapped classifier is a Tf model
        """
        return type(self.clf).__name__ == 'TfMicro'

    def create_for_dataset(self, X, y):
        """
        Create an instance of the classifier
        :param X:
        :param y:
        """
        return self.generator(X=X, y=y)

    def fit(self, X, y, **kwargs):
        """
        Fit classifier
        :param X:
        :param y:
        """
        self.clf = self.create_for_dataset(X, y)

        if self.is_tf() and 'validation_size' in kwargs:
            X, x_validate, y, y_validate = train_test_split(X, y, test_size=kwargs.get('validation_size'))
            kwargs.setdefault('validation_data', (x_validate, y_validate))

        kwargs.pop('validation_size')

        self.clf.fit(X, y, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Run prediction
        """
        return self.clf.predict(*args, **kwargs)

    def port(self, **kwargs):
        """
        Port classifier to C++
        """
        return self.clf.port(**kwargs) if self.is_tf() else micromlgen.port(self.clf, **kwargs)

    def cross_val_score(self, dataset, num_folds, validation_size=0):
        """
        Compute cross validation accuracy
        :param dataset: Dataset
        :param num_folds: int
        :param validation_size: float
        :return: float cross validation score
        """
        assert isinstance(dataset, Dataset), 'dataset MUST be an instance of Dataset'
        assert num_folds > 1, 'num_fold MUST be greather than 1'
        assert 0 <= validation_size < 1, 'validation_size MUST be in the range (0, 1)'

        kfold = KFold(n_splits=num_folds, shuffle=True)
        scores = []

        for train_idx, test_idx in kfold.split(dataset.X, dataset.y):
            self.clf = clf = self.create_for_dataset(dataset.X, dataset.y)
            x_train = dataset.X[train_idx]
            y_train = dataset.y[train_idx]
            x_test = dataset.X[test_idx]
            y_test = dataset.y[test_idx]

            if self.is_tf():
                y_train = to_categorical(y_train)
                if validation_size > 0:
                    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=validation_size)
                    clf.fit(x_train, y_train, validation_data=(x_validate, y_validate))
                else:
                    clf.fit(x_train, y_train)
            else:
                clf.fit(x_train, y_train)

            try:
                score = clf.evaluate(x_test, y_test)
            except AttributeError:
                score = (clf.predict(x_test) == y_test).sum() / len(y_test)

            scores.append(score)

        return np.asarray(scores).mean()

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
