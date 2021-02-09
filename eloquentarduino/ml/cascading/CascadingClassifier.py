import numpy as np
from micromlgen import port
from eloquentarduino.utils import jinja
from eloquentarduino.ml.data.preprocessing import RollingWindow


class CascadingClassifier:
    """
    Use the output of one classifier as input for a second classifier
    """
    def __init__(self, simplex_clf, complex_clf, depth):
        """
        :param simplex_clf: the trained classifier that classifies the primitives
        :param complex_clf: the untrained classifier that will classify the composites
        :param depth: how many primitives make up a composite
        """
        assert simplex_clf is not None and hasattr(simplex_clf, 'predict'), 'simplex_clf MUST implement predict()'
        assert complex_clf is not None and hasattr(complex_clf, 'fit') and hasattr(complex_clf, 'predict'), 'complex_clf MUST implement fit() and predict()'
        assert depth > 1, 'depth MUST be an integer > 1'

        self.simplex_clf = simplex_clf
        self.complex_clf = complex_clf
        self.depth = int(depth)
        self.window = RollingWindow(depth=self.depth)
        self.input_dim = None
        self.classmap = {}
        # make compatible with sklearn classifiers
        self._estimator_type = 'classifier'

    def fit(self, train):
        """
        Fit the complex classifier
        :param train: list of (X, y0), where X is an array of shape num_samples * num_features, y0 is the class idx
        :return: self
        """
        X, y = [], []

        for Xi, y0, *label in train:
            self.input_dim = Xi.shape[1]
            Xi = self.transform(Xi)
            yi = np.ones(len(Xi)) * y0
            X.append(Xi)
            y.append(yi)
            if len(label) == 1:
                self.classmap[int(y0)] = label[0]

        X = np.vstack(X)
        y = np.concatenate(y)

        self.complex_clf.fit(X, y)

        return self

    def transform(self, X):
        """
        Transform input for prediction
        :param X: array of samples to transform
        :return: np.ndarray of predictions
        """
        simplex_y = self.simplex_clf.predict(X).reshape((-1, 1))
        return self.window.transform(simplex_y)

    def predict(self, X):
        """
        Run prediction
        :param X: array of samples to predict
        :return: np.ndarray of predictions
        """
        # accept both "raw" and already transformed data
        if X.shape[1] == self.input_dim:
            X = self.transform(X)
        return self.complex_clf.predict(X)

    def port(self, **kwargs):
        """
        Port to C++
        :return: str C++ code
        """
        classname = kwargs.get('classname', 'CascadingClassifier')
        simplex_classname = '%s_SimplexClassifier' % classname
        complex_classname = '%s_ComplexClassifier' % classname

        return jinja('cascading/CascadingClassifier.jinja', {
            'classname': classname,
            'classmap': self.classmap,
            'simplex_classname': simplex_classname,
            'complex_classname': complex_classname,
            'simplex_clf': port(self.simplex_clf, classname=simplex_classname),
            'complex_clf': port(self.complex_clf, classname=complex_classname),
            'depth': self.depth,
        })
