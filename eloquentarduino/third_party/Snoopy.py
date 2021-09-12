import re
from sklearn.model_selection import cross_validate
from micromlgen import port
from eloquentarduino.utils import jinja
from eloquentarduino.ml.data.preprocessing import RollingWindow
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



class Snoopy:
    def __init__(self):
        self.dataset = None
        self.test_dataset = None
        self.clf = None
        self.config = {
            'depth': None,
            'diff': False,
            'persist': 'false',
            'predict_every': 4
        }
        self.set_voting(5, 5, 0.7, 0.7)

    def set_dataset(self, dataset):
        """
        Set dataset for training
        :param dataset: Dataset
        """
        self.dataset = dataset

    def set_test_dataset(self, dataset):
        """
        Set dataset for testing
        :param dataset: Dataset
        """
        self.test_dataset = dataset

    def diff(self):
        """
        Apply diff() on datasets
        """
        self.config['diff'] = True
        self.dataset = self.dataset.diff()

        if self.test_dataset is not None:
            self.test_dataset = self.test_dataset.diff()

    def rolling_window(self, depth, shift=1):
        """
        Apply rolling window
        :param depth: int rolling window depth
        :param shift: int rolling window shift
        """
        def f(X):
            return RollingWindow(depth=depth, shift=shift).transform(X, flatten=True)

        self.config['depth'] = depth
        self.config['shift'] = shift
        self.dataset.transform_splits(f)

        if self.test_dataset is not None:
            self.test_dataset.transform_splits(f)

    def set_classifier(self, clf, cv=3):
        """
        Train classifier
        :param clf: Classifier
        :param cv: int cross validation splits
        """
        assert self.dataset is not None, 'you MUST set a dataset first'

        X, y = self.dataset.Xy_shuffle
        scores = cross_validate(clf, X, y, cv=cv, return_estimator=True)
        best_idx = scores['test_score'].argmax()
        self.clf = scores['estimator'][best_idx]
        self.clf.fit(X, y)

        return scores['test_score'][best_idx]

    def get_test_y(self):
        """

        """
        if self.test_dataset is None:
            return None

        return self.clf.predict(self.test_dataset.X)

    def test_score(self):
        """
        Get score on test dataset
        :return: float accuracy
        """
        if self.test_dataset is None:
            return 0

        return self.clf.score(self.test_dataset.X, self.test_dataset.y)

    def set_voting(self, short_term, long_term, short_quorum=0.7, long_quorum=0.7):
        """
        Set short-long term voting scheme
        """
        self.config['voting'] = (short_term, long_term, short_quorum, long_quorum)

    def set_frequency(self, n):
        """
        Set prediction frequency
        """
        self.config['predict_every'] = n

    def set_persistance(self, persist):
        """

        """
        self.config['persist'] = 'true' if persist else 'false'

    def set_project(self, project):
        """
        Export class to Arduino sketch
        """
        ported_clf = port(self.clf, classname='Classifier', classmap=self.dataset.classmap, pretty=True)
        self.config.update(ported_clf=ported_clf, num_features=len(self.dataset.df.columns))

        contents = jinja('third_party/snoopy/snoopy.jinja', self.config)
        project.files.add('ML.h', contents=contents, exists_ok=True)

    def plot_confusion_matrix(self, test_dataset=None, normalize='true', cmap='viridis', xticks_rotation=45, include_values=True, **kwargs):
        """
        Plot confusion matrix
        """
        if test_dataset is None:
            test_dataset = self.test_dataset or self.dataset

        y_true = test_dataset.y
        y_pred = self.clf.predict(test_dataset.X)
        classmap = test_dataset.classmap

        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[label for i, label in classmap.items()])
        disp.plot(include_values=include_values, cmap=cmap, xticks_rotation=xticks_rotation)
