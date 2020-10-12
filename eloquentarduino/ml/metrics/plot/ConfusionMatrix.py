import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class ConfusionMatrix:
    """
    Plotter for confusion matrix
    """
    def __init__(self, cf):
        self._cf = cf
        self._cmap = "Blues"
        self._range = (0, 1)

    def cmap(self, cmap):
        """
        Set colormap
        :param cmap:
        :return:
        """
        self._cmap = cmap
        return self

    def range(self, range):
        """
        Set values range
        :param range:
        :return:
        """
        assert isinstance(range, tuple) and len(range) == 2, 'range MUST be a pair'
        self._range = range
        return self

    def plot(self, ax=plt, label='', annot_kws={}):
        """
        Plot confusion matrix
        :param ax:
        :param annot_kws:
        :return:
        """
        default_annot_kws = {"size": 16}
        default_annot_kws.update(annot_kws)
        num_classes = len(self._cf)
        features = [str(i) for i in range(num_classes)]
        df = pd.DataFrame(self._cf, index=features, columns=features)
        ax = sns.heatmap(df,
                         annot=True,
                         annot_kws=default_annot_kws,
                         cmap=self._cmap,
                         ax=ax,
                         vmin=self._range[0],
                         vmax=self._range[1])
        ax.set_xlabel('Predicted label\n%s' % label)
        ax.set_ylabel('True label')
        plt.show()