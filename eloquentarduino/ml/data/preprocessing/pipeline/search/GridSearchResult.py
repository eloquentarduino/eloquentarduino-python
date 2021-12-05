import numpy as np
from cached_property import cached_property
from eloquentarduino.plot import ConfusionMatrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class GridSearchResult(dict):
    """
    Add syntactic sugar to pipeline grid search results
    """
    @cached_property
    def score(self):
        """
        Get score
        :return: float
        """
        return self["score"]

    @cached_property
    def accuracy_score(self):
        """
        Get accuracy score
        :return: float
        @added 0.1.19
        """
        return accuracy_score(self["y_true"], self["y_pred"])

    @cached_property
    def precision_score(self):
        """
        Get precision score
        :return: float
        @added 0.1.19
        """
        return precision_score(self["y_true"], self["y_pred"], average='macro')

    @cached_property
    def recall_score(self):
        """
        Get recall score
        :return: float
        @added 0.1.19
        """
        return recall_score(self["y_true"], self["y_pred"], average='macro')

    @cached_property
    def f1_score(self):
        """
        Get F1 score
        :return: float
        @added 0.1.19
        """
        return f1_score(self["y_true"], self["y_pred"], average='macro')

    @cached_property
    def missing_rate(self):
        """
        Get missing rate of pipeline
        :return: float missing rate from 0 to 1
        """
        inRow = self["pipeline"]["InRow"]

        return 0 if inRow is None else inRow.missing_rate

    def print_scores(self):
        """
        Print all scores
        """
        print("Score    : %.2f" % self["score"])
        print("Accuracy : %.2f" % self.accuracy_score)
        print("Precision: %.2f" % self.precision_score)
        print("Recall   : %.2f" % self.recall_score)
        print("F1       : %.2f" % self.f1_score)
        print("Missing rate: %.2f" % self.missing_rate)

    def get_accuracy_score_of_classes(self, classes):
        """
        Get result accuracy over a subset of classes
        @added 0.1.19
        """
        return self.get_masked_score(accuracy_score, mask=np.isin(self['y_true'], classes))
    
    def get_precision_score_of_classes(self, classes):
        """
        Get result precision over a subset of classes
        @added 0.1.19
        """
        return self.get_masked_score(
            lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
            mask=np.isin(self['y_true'], classes))
    
    def get_recall_score_of_classes(self, classes):
        """
        Get result recall over a subset of classes
        @added 0.1.19
        """
        return self.get_masked_score(
            lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
            mask=np.isin(self['y_true'], classes))
    
    def get_f1_score_of_classes(self, classes):
        """
        Get result F1 over a subset of classes
        @added 0.1.19
        """
        return self.get_masked_score(
            lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
            mask=np.isin(self['y_true'], classes))
    
    def get_masked_score(self, score_function, mask):
        """
        Get result score over the masked predictions
        :param score_function: callable
        :param mask: numpy.ndarray
        :return: float
        @added 0.1.19
        """
        return score_function(self['y_true'][mask], self['y_pred'][mask])

    def plot_confusion_matrix(self, labels=None, **kwargs):
        """
        Plot confusion matrix of results
        :param labels: list or None
        """
        ConfusionMatrix(self["y_true"], self["y_pred"], labels=labels).show(**kwargs)

    def plot_all_confusion_matrices(self, labels=None, **kwargs):
        """
        Plot confusion matrices of results with different normalizations
        :param labels: list or None
        """
        cm = ConfusionMatrix(self["y_true"], self["y_pred"], labels=labels)

        cm.show(title="Raw values", normalize=None, **kwargs)
        cm.show(title="Norm=true (recall)", normalize="true", **kwargs)
        cm.show(title="Norm=pred (precision)", normalize="pred", **kwargs)
