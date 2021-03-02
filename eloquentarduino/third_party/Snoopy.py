import re
from sklearn.model_selection import cross_validate
from micromlgen import port
from eloquentarduino.utils import jinja
from eloquentarduino.ml.data.preprocessing import RollingWindow



class Snoopy:
    def __init__(self):
        self.dataset = None
        self.clf = None
        self.config = {
            'depth': None,
            'diff': False,
            'persist': 'false',
            'predict_every': 4
        }
        self.set_voting(5, 5, 0.7, 0.7)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def rolling_window(self, depth, shift=1, diff=True):
        def f(X):
            return RollingWindow(depth=depth, shift=shift).transform(X, flatten=True)

        if diff:
            self.config['diff'] = True
            self.dataset = self.dataset.diff()

        self.config['depth'] = depth
        self.dataset.transform_splits(f)

    def set_classifier(self, clf, cv=3):
        assert self.dataset is not None, 'you MUST set a dataset first'

        scores = cross_validate(clf, self.dataset.X, self.dataset.y, cv=cv, return_estimator=True)
        best_idx = scores['test_score'].argmax()
        self.clf = scores['estimator'][best_idx]
        self.clf.fit(self.dataset.X, self.dataset.y)

        return scores['test_score'][best_idx]

    def set_voting(self, short_term, long_term, short_quorum=0.7, long_quorum=0.7):
        self.config['voting'] = (short_term, long_term, short_quorum, long_quorum)

    def set_frequency(self, n):
        self.config['predict_every'] = n

    def set_persistance(self, persist):
        self.config['persist'] = 'true' if persist else 'false'

    def set_project(self, project):
        ported_clf = port(self.clf, classname='Classifier', classmap=self.dataset.classmap, pretty=True)
        self.config.update(ported_clf=ported_clf, num_features=len(self.dataset.df.columns))

        contents = jinja('third_party/snoopy/snoopy.jinja', self.config)
        project.files.add('ML.h', contents=contents, exists_ok=True)
