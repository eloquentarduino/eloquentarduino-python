import numpy as np
from eloquentarduino.ml.data.loaders import rolling_window
from sklearn.model_selection import cross_val_score
from eloquentarduino.utils import jinja
from micromlgen import port


class CascadingClassifier:
    """
    Use the output of one classifier as input for a second classifier
    """
    def __init__(self):
        self.primitives_dataset = None
        self.composites_dataset = None
        self.primitives_classifier = None
        self.composites_classifier = None
        self.feature_extractor = None
        self.composites_length = None
        self.primitives_summary_axis = None
        self.X = None
        self.y = None
        self.X_primitives = None
        self.y_primitives = None
        self.X_composites = None
        self.y_composites = None

    def set_primitives_dataset(self, X, y, classmap=None):
        """
        Set primitives dataset (used to learn elementary patterns)
        :param X: features
        :param y: labels
        :param classmap:
        """
        self.primitives_dataset = [X, y, classmap]
        return self

    def set_composites_dataset(self, X, y, classmap=None):
        """
        Set composites dataset (used to learn complex patterns)
        :param X: features
        :param y: labels
        :param classmap:
        """
        self.composites_dataset = [X, y, classmap]
        return self

    def set_preprocessing(self, feature_extractor):
        """
        Set function to preprocess data and extract features
        :param feature_extractor: function
        """
        self.feature_extractor = feature_extractor
        return self

    def set_primitives_classifier(self, clf):
        """
        Set classifier for primitives patterns
        :param clf:
        """
        self.primitives_classifier = clf
        return self

    def set_composites_classifier(self, clf):
        """
        Set classifier for complex patterns
        :param clf:
        """
        self.composites_classifier = clf
        return self

    def set_composites_length(self, length):
        """
        Set how many primitives to consider for classifying a composite
        :param length: number of primitives that form a composite
        """
        self.composites_length = length
        return self

    def set_primitives_summary_axis(self, axis):
        """
        Set axis along which compute primitives' features statistics
        :param axis: if axis=1, you compute summary along each feature (produces k * N features)
                     if axis=2, you compute summary along each window (produces k * W features)
                     if axis=(1, 2), you compute summary along the whole window (produces k features)
                     'featurewise' is an alias for 1
                     'windowwise' is an alias for 2
                     'global' is an alias for (1, 2)
        :return: self
        """
        if axis == 'featurewise':
            axis = 1
        elif axis == 'windowwise':
            axis = 2
        elif axis == 'global':
            axis = (1, 2)
        assert axis is None or axis == 1 or axis == 2 or axis == (1, 2), 'axis MUST be one of {None, 1, 2, (1, 2)}'
        self.primitives_summary_axis = axis
        return self

    def fit(self, cv=0):
        """
        Train the two cascading classifiers
        :param cv: cross validation splits
        :return: the cross validation scores, if any
        """
        assert self.primitives_dataset is not None, 'use set_primitives_dataset(X, y, classmap) to set a primitives dataset'
        assert self.composites_dataset is not None, 'use set_composites_dataset(X, y, classmap) to set a composites dataset'
        assert self.primitives_classifier is not None, 'use set_primitives_classifier(clf) to set a primitives classifier'
        assert self.composites_classifier is not None, 'use set_composites_classifier(clf) to set a composites classifier'
        assert self.composites_length is not None and self.composites_length > 1, 'use set_composites_length(length) to set a composites length greater than 1'

        if self.feature_extractor is not None:
            self.primitives_dataset[:2] = self.feature_extractor(*self.primitives_dataset[:2])
            self.composites_dataset[:2] = self.feature_extractor(*self.composites_dataset[:2])

        X_primitives, y_primitives, classmap_primitives = self.primitives_dataset
        X_composites, y_composites, classmap_composites = self.composites_dataset
        X, y = [], []

        self.primitives_classifier.fit(X_primitives, y_primitives)

        for yi in np.unique(y_composites):
            composite = X_composites[y_composites == yi]
            y_pred = self.primitives_classifier.predict(composite)
            y_pred = self._make_windows(y_pred)
            features = self._make_features(composite)
            features = self._concatenate(features, y_pred)
            X += features.tolist()
            y += [yi for i in range(len(features))]

        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.X_primitives = X_primitives
        self.X_composites = X_composites
        self.y_primitives = y_primitives
        self.y_composites = y_composites
        scores = np.sort(cross_val_score(self.composites_classifier, self.X, self.y, cv=cv)) if cv > 1 else None
        self.composites_classifier.fit(self.X, self.y)

        return scores

    def transform(self, X, extract=True):
        """
        Transform input
        :param X: dataset
        :param extract: apply feature extractor
        :return: np.ndarray predictions
        """
        if extract and self.feature_extractor is not None:
            y = np.zeros(len(X))
            X, _ = self.feature_extractor(X, y)

        y_pred = self.primitives_classifier.predict(X)
        y_pred = self._make_windows(y_pred)
        features = self._make_features(X)

        return self._concatenate(features, y_pred)

    def predict(self, X, extract=True):
        """
        Run inference
        :param X: dataset
        :param extract: apply feature extractor
        :return: np.ndarray predictions
        """
        return self.composites_classifier.predict(self.transform(X, extract=extract))

    def port(self, pretty=True, **kwargs):
        """
        Port to plain C++
        :param pretty: turn on pretty print
        :return: str plain C++ code
        """
        env = {
            'num_features_primitives': self.X_primitives.shape[1],
            'num_features': self.X.shape[1],
            'axis': self.primitives_summary_axis,
            'composites_length': self.composites_length,
            'clf_primitives': port(self.primitives_classifier, classname='PrimitivesClassifier', classmap=self.primitives_dataset[2], **kwargs),
            'clf_composites': port(self.composites_classifier, classname='CompositesClassifier', classmap=self.composites_dataset[2], **kwargs),
        }
        return jinja("CascadingClassifier/CascadingClassifier.jinja", env, pretty=pretty)

    def _make_windows(self, array):
        """
        Apply rolling window of composite length
        :param array:
        :return: np.ndarray
        """
        windows = rolling_window(array, window=self.composites_length, overlap=self.composites_length - 1)
        keep_mask = [i for i, w in enumerate(windows) if len(w) == self.composites_length]

        return windows[keep_mask]

    def _make_features(self, X):
        """
        Extract features from data
        :param X:
        :return: np.ndarray
        """
        if self.primitives_summary_axis is None:
            return None

        ax = self.primitives_summary_axis
        composite_windows = self._make_windows(X)
        min_ = composite_windows.min(axis=ax)
        max_ = composite_windows.max(axis=ax)
        mean_ = composite_windows.mean(axis=ax)
        std_ = composite_windows.var(axis=ax)

        if len(min_.shape) == 1:
            min_ = min_.reshape((-1, 1))
            max_ = max_.reshape((-1, 1))
            mean_ = mean_.reshape((-1, 1))
            std_ = std_.reshape((-1, 1))

        return np.hstack((min_, max_, mean_, std_))

    def _concatenate(self, features, y_pred):
        """
        Concatenate arrays if not None
        :param features:
        :param y_pred:
        :return: np.ndarray contatenation
        """
        return np.hstack((features, y_pred)) if features is not None else y_pred
