import numpy as np
from eloquentarduino.ml.data.loaders import rolling_window
from sklearn.model_selection import cross_val_score


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
            windows = rolling_window(y_pred, window=self.composites_length, overlap=self.composites_length-1)
            windows = [w for w in windows if len(w) == self.composites_length]
            X += windows
            y += [yi for i in range(len(windows))]

        scores = np.sort(cross_val_score(self.composites_classifier, np.asarray(X), np.asarray(y), cv=cv)) if cv > 1 else None
        self.composites_classifier.fit(np.asarray(X), np.asarray(y))

        return scores
