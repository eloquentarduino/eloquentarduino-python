import matplotlib.pyplot as plt
import seaborn as sns


class BenchmarkPlotter:
    """
    Plot benchmark results
    """
    def __init__(self, df):
        """
        :param df: benchmark results as DataFrame
        """
        self.df = df

    def size(self, size, dpi=200):
        """
        Set matplolib defaults
        :param size:
        :param dpi:
        :return:
        """
        plt.rcParams['figure.figsize'] = size
        plt.rcParams['figure.dpi'] = dpi

    def bar(self, x, y, groupby, divideby=None, title=None, yscale=None, df=None):
        """
        Plot bars
        :param x:
        :param y: column or computed property
        :param groupby:
        :param divideby:
        :param title:
        :param yscale:
        :param df: for internal use only
        :return: sns.FacetGrid
        """
        df, y = self.prepare(df, y)

        # draw a single plot for each label
        if divideby is not None:
            return [
                self.bar(x=x, y=y, groupby=groupby, title=[divide, title], yscale=yscale, df=df[df[divideby] == divide])
                for divide in df[divideby].unique()
            ]

        g = sns.catplot(data=df.sort_values(by=y), x=groupby, y=y, hue=x, kind='bar')
        g.set_xticklabels(rotation=80)
        self.add_title(g, title)

        if yscale:
            g.set(yscale=yscale)

        return g

    def scatter(self, x, y, hue=None, marker=None, title=None, xscale=None, yscale=None, df=None):
        df, y = self.prepare(df, y)
        df, xscaler = self.prepare_scale(df, x, {'xscale': xscale})
        df, yscaler = self.prepare_scale(df, y, {'yscale': yscale})
        g = sns.scatterplot(data=df, x=x, y=y, hue=hue, style=marker)
        plt.setp(g.get_legend().get_texts(), fontsize='8')

        # self.add_title(g, title)
        xscaler(g)
        yscaler(g)

        return g

    def prepare(self, df, y):
        """
        Parse arguments
        :param df:
        :param y:
        :return:
        """
        if df is None:
            df = self.df

        # allow computable y
        if callable(y):
            df[y.__name__] = y(df)
            y = y.__name__

        return df, y

    def add_title(self, g, title):
        """
        Add title to plot
        :param g:
        :param title:
        :return:
        """
        if title is not None:
            # allow 2-levels titles
            if isinstance(title, list):
                title = ' » '.join([x for x in title if x])

            plt.subplots_adjust(top=0.9)
            g.fig.suptitle(title)

    def prepare_scale(self, df, column, scale):
        scaler = lambda g: None
        scale_value = [v for v in scale.values()][0]

        if column is not None and scale_value is not None:
            values = df[column]
            scaler = lambda g: g.set(**scale)

            # move xs from range [0, 1] to range [1, 2]
            if scale_value == 'log' and values.min() < 1:
                df[column] += 1 - values.min()

        return df, scaler

