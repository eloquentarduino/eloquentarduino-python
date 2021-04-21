import re
from eloquentarduino.ml.data.preprocessing.pipeline import Pipeline
from eloquentarduino.ml.classification.abstract.Classifier import Classifier
from eloquentarduino.utils import jinja


class SnoopyStream:
    """
    SnoopyStream generator
    """
    @staticmethod
    def from_grid_search(result):
        """
        Static constructor
        """
        return SnoopyStream(**result)

    def __init__(self, pipeline, clf, voting, **kwargs):
        assert isinstance(pipeline, Pipeline), 'pipeline MUST be a Pipeline instance'
        assert isinstance(clf, Classifier), 'clf MUST be a Classifier instance'
        assert isinstance(voting, dict) and 'short' in voting and 'long' in voting and 'quorum' in voting, 'voting MUST be in the form {short: x, long: x, quorum: x}'

        self.pipeline = pipeline
        self.clf = clf
        self.voting = voting

    def port(self, persist=False):
        return jinja('third_party/snoopy/SnoopyStream.jinja', {
            'pipeline_ns': self.pipeline.name,
            'pipeline': self.uglify(self.pipeline.port(classname='Pipeline')),
            'clf': self.uglify(self.clf.port(classname='Classifier')),
            'voting': self.voting,
            'persist': 'true' if persist else 'false'
        })

    def uglify(self, code):
        # drop comments
        code = re.sub(r'//[^\n]+\n', '\n', code)
        # make a one-liner
        code = code.replace('#pragma once', '')
        code = re.sub(r'\n+', '   ', code).strip()
        # lines starting with # deserve their one line
        code = re.sub('((#define|#include).+?)   ', lambda m: m.group(0) + '\n', code)
        return code


