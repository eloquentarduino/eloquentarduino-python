import re
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