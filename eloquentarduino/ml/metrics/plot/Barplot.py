import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Barplot:
    """
    Draw a barplot
    """
    def __init__(self, df, x, y, compare=None, factor=None, sort=False, scale='linear', **kwargs):
        """
        Create a barplot
        :param df:
        :param x:
        :param y:
        """
        if sort:
            df = df.sort_values(by=y)

        if compare is None:
            kwargs.setdefault('palette', self.palette(df[y]))
            g = sns.barplot(data=df, x=x, y=y, **kwargs)

            if scale == 'log':
                g.set_yscale('log')
        else:
            df = pd.concat((df, compare.df))
            kwargs.setdefault('palette', ['#345f44', '#308b55'])
            g = sns.catplot(data=df, x=x, y=y, hue=factor, kind='bar', **kwargs)

            if scale == 'log':
                g.set(yscale='log')

        plt.xticks(rotation=30)
        plt.show()

    def palette(self, ys):
        """
        Generate custom palette
        :param ys:
        :return:
        """
        # @from https://stackoverflow.com/questions/36271302/changing-color-scale-in-seaborn-bar-plot#answer-60917129
        palette = sns.color_palette("Greens_d", len(ys))

        if len(ys) > 1 and len([y for y in ys if y != 0]) > 0:
            normalized = (ys - min(ys)) / (max(ys) - min(ys))
            indices = np.round(normalized * (len(ys) - 1)).astype(np.int32)
            return np.array(palette).take(indices, axis=0)

        return palette