import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class ConfusionMatrix:
    """
    Plotter for confusion matrix
    """
    def __init__(self, cf, compare=None):
        if compare is None:
            self.cf = cf
            self.cmap = "Greens"
            self.range = (0, 1)
        else:
            self.cf = cf - compare
            self.cmap = "vlag_r"
            self.range = (-1, 1)

    def plot(self, label='', annot_kws={}):
        """
        Plot confusion matrix
        :param label:
        :param annot_kws:
        :return:
        """
        default_annot_kws = {"size": 16}
        default_annot_kws.update(annot_kws)
        num_classes = len(self.cf)
        features = [str(i) for i in range(num_classes)]
        df = pd.DataFrame(self.cf, index=features, columns=features)
        fig = plt.figure(figsize=(num_classes * 2, num_classes))
        ax = fig.add_subplot(111)
        ax = sns.heatmap(df,
                         annot=True,
                         annot_kws=default_annot_kws,
                         cmap=self.cmap,
                         ax=ax,
                         vmin=self.range[0],
                         vmax=self.range[1])
        ax.set_xlabel('Predicted label\n%s' % label)
        ax.set_ylabel('True label')
        plt.show()