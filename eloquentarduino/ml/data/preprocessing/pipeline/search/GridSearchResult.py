from eloquentarduino.plot import ConfusionMatrix


class GridSearchResult(dict):
    """
    Add synctactic sugar to pipeline grid search results
    """
    @property
    def score(self):
        """
        Get score
        :return: float
        """
        return self["score"]

    @property
    def missing_rate(self):
        """
        Get missing rate of pipeline
        :return: float missing rate from 0 to 1
        """
        inRow = self["pipeline"]["InRow"]

        return 0 if inRow is None else inRow.missing_rate

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
