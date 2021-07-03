from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep
from eloquentarduino.ml.data.preprocessing.pipeline.Window import Window
#from eloquentarduino.ml.data.preprocessing.pipeline.Pipeline import Pipeline
from eloquentarduino.ml.classification.abstract.Classifier import Classifier


class Cascading(BaseStep):
    """
    Perform cascading classification
    """
    def __init__(self, pipeline, name='Cascading'):
        """
        :param simple_pipeline: Pipeline
        :param clf: Classifier
        """
        super().__init__(name=name)
        self.pipeline = pipeline
        self.simplex_clf = None
        self.complex_clf = None
        self.window = None
        self.Xw = None
        self.yw = None

    def window_length(self, length):
        """
        Set window size
        :param length: int
        """
        assert length > 1, 'window MUST be greater than 1'

        self.window = Window(length=length, flatten=True, name='CascadingWindow')

        return self

    def simplex(self, clf):
        """
        Set simplex classifier
        :param clf: Classifier
        """
        assert isinstance(clf, Classifier), 'clf MUST be a ml.classification.abstract.Classifier instance'

        self.simplex_clf = clf

        return self

    def complex(self, clf):
        """
        Set complex classifier
        :param clf: Classifier
        """
        assert isinstance(clf, Classifier), 'clf MUST be a ml.classification.abstract.Classifier instance'

        self.complex_clf = clf

        return self

    def fit(self, X, y):
        """
        Fit classifier on simple_pipeline's output
        """
        assert self.window is not None, 'you MUST call window_length()'
        assert self.simplex_clf is not None, 'you MUST call simplex()'
        assert self.complex_clf is not None, 'you MUST call complex()'

        if self.pipeline.X is None:
            self.pipeline.fit()

        self.set_X(X)
        self.simplex_clf.fit(self.pipeline.X, self.pipeline.y)

        X_train, y_train = self.pipeline.transform(X, y)
        y_pred = self.simplex_clf.predict(X_train)
        X_w, y_w = self.window.fit(y_pred.reshape((-1, 1)), y_train)

        self.Xw = X_w
        self.yw = y_w

        return self.complex_clf.fit(X_w, y_w).predict(X_w).reshape((-1, 1)), y_w

    def transform(self, X, y=None):
        """
        Return prediction on simple_pipeline's output
        """
        assert self.window is not None, 'you MUST call window_length()'
        assert self.simplex_clf is not None, 'you MUST call simplex()'
        assert self.complex_clf is not None, 'you MUST call complex()'

        X_t, y_t = self.pipeline.transform(X, y)
        y_pred = self.simplex_clf.predict(X_t)
        X_w, y_w = self.window.transform(y_pred.reshape((-1, 1)), y_t)

        return self.complex_clf.predict(X_w).reshape((-1, 1)), y_w

    def get_template_data(self):
        """

        """
        return {
            'pipeline': self.pipeline.port(classname='SimplexPipeline'),
            'simplex_clf': self.simplex_clf.port(classname='SimplexClassifier'),
            'complex_clf': self.complex_clf.port(classname='ComplexClassifier'),
            'window': self.window.port(ns=self.pipeline.name),
            'simple_ns': self.pipeline.name,
            'window_length': self.window.length
        }

