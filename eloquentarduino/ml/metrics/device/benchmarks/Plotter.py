import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


class Plotter:
    """
    Plot benchmark results
    """
    def __init__(self, results):
        """
        Constructor
        :param results: string|list|DataFrame the results to plot. Can either be a path to a file, a list of dicts or a pandas Dataframe
        """
        assert isinstance(results, str) or isinstance(results, list) or isinstance(results, DataFrame), 'results MUST be either a string, a list of dict or a DataFrame'

        if isinstance(results, str):
            self.df = pd.read_csv(results)
        elif isinstance(results, DataFrame):
            self.df = results
        else:
            self.df = DataFrame(results)

        self.size = (None, None)

    def figure(self):
        """
        Create new figure
        """
        figsize, dpi = self.size

        plt.figure(figsize=figsize, dpi=dpi)

    def set_size(self, size, dpi=200):
        """
        Set plot size and dpi
        :param size: tuple
        :param dpi: int
        :return:
        """
        self.size = (size, dpi)

    def where(self, key, value=None):
        """
        Filter inner dataframe
        :param key: callable|string
        :param value: mixed
        :return: Plotter a new plotter on the filtered data
        """
        assert callable(key) or value is not None, 'you MUST supply a callable or a (key, value) pair'

        df = key(self.df) if callable(key) else self.df[self.df[key] == value]

        plotter = Plotter(df)
        plotter.set_size(*self.size)

        return plotter

    def plot_accuracy(self, groupby='dataset', divideby='board', hueby='clf', scale=None):
        """
        Plot accuracy
        :param groupby: string which dimension to plot on the x axis
        :param divideby: string which dimension to create each chart for
        :param hueby: string which dimension to color the bars by
        :param scale: string scale for y axis
        """
        return self.bar(x=groupby, y='accuracy', divideby=divideby, hueby=hueby)

    def plot_flash(self, groupby='dataset', divideby='board', hueby='clf', scale='log'):
        """
        Plot flash
        :param groupby: string which dimension to plot on the x axis
        :param divideby: string which dimension to create each chart for
        :param hueby: string which dimension to color the bars by
        :param scale: string scale for y axis
        """
        return self.bar(x=groupby, y='flash', divideby=divideby, hueby=hueby, y_scale=scale)

    def plot_memory(self, groupby='dataset', divideby='board', hueby='clf', scale='log'):
        """
        Plot memory
        :param groupby: string which dimension to plot on the x axis
        :param divideby: string which dimension to create each chart for
        :param hueby: string which dimension to color the bars by
        :param scale: string scale for y axis
        """
        return self.bar(x=groupby, y='memory', divideby=divideby, hueby=hueby, y_scale=scale)

    def plot_accuracy_by_flash(self, hueby='clf', markerby='dataset'):
        """
        Plot accuracy by flash
        :param hueby: string which dimension to color the points by
        :param markerby: string which dimension to style the points by
        :return:
        """
        return self.scatter(x='flash', y='accuracy', hue=hueby, marker=markerby, x_scale='log')

    def plot_accuracy_by_memory(self, hueby='clf', markerby='dataset'):
        """
        Plot accuracy by memory
        :param hueby: string which dimension to color the points by
        :param markerby: string which dimension to style the points by
        :return:
        """
        return self.scatter(x='memory', y='accuracy', hue=hueby, marker=markerby, x_scale='log')

    def plot_accuracy_by_inference_time(self, hueby='clf', markerby='dataset'):
        """
        Plot accuracy by inference time
        :param hueby: string which dimension to color the points by
        :param markerby: string which dimension to style the points by
        :return:
        """
        return self.scatter(x='inference_time', y='accuracy', hue=hueby, marker=markerby, x_scale='log')

    def plot_inference_time_by_flash(self, hueby='clf', markerby='dataset'):
        """
        Plot inference time by flash
        :param hueby: string which dimension to color the points by
        :param markerby: string which dimension to style the points by
        :return:
        """
        return self.scatter(x='flash', y='inference_time', hue=hueby, marker=markerby, x_scale='log', y_scale='log')

    def plot_inference_time_by_memory(self, hueby='clf', markerby='dataset'):
        """
        Plot inference time by memory
        :param hueby: string which dimension to color the points by
        :param markerby: string which dimension to style the points by
        :return:
        """
        return self.scatter(x='memory', y='inference_time', hue=hueby, marker=markerby, x_scale='log', y_scale='log')

    def bar(self, x, y, hueby=None, divideby=None, y_scale=None):
        """
        Bar plot
        :param x: string which dimension to plot on the x axis
        :param y: string which dimension to plot on the y axis
        :param divideby: string which dimension to create each chart for
        :param hueby: string which dimension to color the bars by
        :param y_scale: string scale for y axis
        """
        divides = self.df[divideby].unique() if divideby is not None else [None]
        plots = []

        for divide in divides:
            self.figure()

            title = '%s » %s by %s' % (divide, y, x) if divide is not None else x
            g = sns.barplot(data=self.df.sort_values(by=y), x=x, y=y, hue=hueby, n_boot=50)

            plt.xticks(rotation=80)
            plt.subplots_adjust(top=0.9)
            g.set_title(title)

            if y_scale is not None:
                g.set_yscale(y_scale)

            plots.append(g)

        return plots

    def scatter(self, x, y, hue=None, marker=None, x_scale=None, y_scale=None):
        """
        Scatter plot
        :param x: string which dimension to plot on the x axis
        :param y: string which dimension to plot on the y axis
        :param hue: string which dimension to color the points by
        :param marker: string which dimension to style the points by
        :param x_scale: string scale for x axis
        :param y_scale: string scale for y axis
        """
        self.figure()

        # if hue or marker have a single value, drill down by something else
        available_drilldowns = [drill for drill in ['clf', 'dataset', 'board'] if drill not in [hue, marker]]

        if hue is not None and len(self.df[hue].unique()) == 1:
            hue = available_drilldowns.pop()

        if marker is not None and len(self.df[marker].unique()) == 1:
            marker = available_drilldowns.pop()

        xs = self.df[x]
        ys = self.df[y]
        x_suffix = '_log' if x_scale == 'log' else ''
        y_suffix = '_log' if y_scale == 'log' else ''

        if x_scale == 'log' and xs.min() < 1:
            xs = np.log(xs + 1)

        if y_scale == 'log' and ys.min() < 1:
            ys = np.log(ys + 1)

        g = sns.scatterplot(data=self.df, x=xs, y=ys, hue=hue, style=marker)

        plt.title('%s%s by %s%s' % (y, y_suffix, x, x_suffix))
        plt.setp(g.get_legend().get_texts(), fontsize='8')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        return g


