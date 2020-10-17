import matplotlib.pyplot as plt
from eloquentarduino.ml.metrics.plot.Series import Series


class Bar(Series):
    """
    Bar plot
    """
    def __init__(self, *args, **kwargs):
        Series.__init__(self, *args, **kwargs)
        self._hat = None

    def hat(self, f_or_list):
        """
        Add hat text to bars
        :param f_or_list: callable or list of texts
        :return: self
        """
        assert callable(f_or_list) or isinstance(f_or_list, list), 'hat MUST be a callable or a list'
        self._hat = f_or_list
        return self

    def plot(self, ax=plt, *args, **kwargs):
        """
        Plot bar
        :param ax:
        :param args:
        :param kwargs:
        :return:
        """
        if len(self.filtered) > 0:
            self._kwargs.update(kwargs)
            bar = ax.bar(self.xs, self.ys, **self._kwargs)
            self._annotate(ax, bar)

    def _annotate(self, ax, bar):
        """
        Add annotations to bar
        :param ax:
        :param bar:
        :return:
        """
        for idx, rect in enumerate(bar):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, self._get_annotation(idx), ha='center', va='bottom', rotation=0)

    def _get_annotation(self, idx):
        """
        Get annotation for given item
        :param idx:
        :return: str
        """
        if isinstance(self._hat, list):
            return self._hat[idx]
        return self._hat(*self.filtered[idx])
