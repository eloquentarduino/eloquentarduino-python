import re
import numpy as np
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep
from eloquentarduino.ml.classification.abstract.Classifier import Classifier


class Classify(BaseStep):
    """
    Apply classification
    """
    def __init__(self, clf, name='Classify'):
        """
        :param clf: Classifier
        """
        assert isinstance(clf, Classifier), 'clf MUST be a ml.classification.abstract.Classifier instance'

        super().__init__(name)
        self.clf = clf

    def fit(self, X, y):
        """
        Fit
        """
        self.set_X(X)
        self.clf.fit(X, y)

        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Transform
        """
        y_pred = self.clf.predict(X)

        # reshape NN predictions (N x M) to 1d array
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)

        return y_pred.reshape((-1, 1)), y

    def get_template_data(self):
        """

        """
        return {
            'clf_code': self.clf.port(classname='Classifier')
        }

    def postprocess_port(self, ported):
        """

        """
        # drop duplicated `#pragma once`
        ported = re.sub(r'(#pragma once[\s\S]+)#pragma once', lambda g: g.group(1), ported)

        return ported

    def get_config(self):
        """
        Show details of classifier
        """
        return {
            'clf': self.clf
        }