import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np


class PlotsItselfMixin:
    """
    Mixin to plot dataset
    Made to be used by Dataset
    """
    is_plot_disabled = False

    @classmethod
    def disable_plots(cls, disabled=True):
        """
        Disable plots (for faster Notebook execution)
        :param disabled: bool if True, no plot is drawn
        """
        cls.is_plot_disabled = disabled

    def plot(self, title='', columns=None, n_ticks=15, grid=True, fontsize=6, bg_alpha=0.2, once_every=1, palette=None, y_pred=None, force=False, **kwargs):
        """
        Plot dataframe
        :param title: str title of plot
        :param columns: list columns to plot
        :param n_ticks: int number of ticks on the x axis
        :param grid: bool wether to display the grid
        :param fontsize: int font size for the axis values
        :param bg_alpha: float alpha of classes' background color
        :param once_every: int limit the number of samples to draw
        :param y_pred: np.array draw predictions markers on top of plot
        :param force: bool if True, always draw the plot, no matter disable_plot() calls
        """
        if not self._should_plot(force):
            print('Dataset plotting is disabled, skipping...')
            return

        plt.figure()
        plot_columns = [c for c in (columns or list(self.df.columns)) if c != 'y']
        df = pd.DataFrame(self.df[plot_columns].iloc[::once_every].to_numpy(), columns=plot_columns)
        length = len(df)

        df.plot(title=title, xticks=range(0, length, length // n_ticks), grid=grid, fontsize=fontsize, rot=70, **kwargs)

        # highlight labels
        y = self.y[::once_every]
        loc_run_start = np.empty(len(y), dtype=bool)
        loc_run_start[0] = True
        np.not_equal(y[:-1], y[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]
        run_lengths = np.diff(np.append(run_starts, len(y)))
        run_values = y[loc_run_start]
        palette = [c for c in (palette or mcolors.TABLEAU_COLORS.values())]

        if self.y.max() >= len(palette):
            print('[WARN] too many classes for the current palette')

        for v, s, l in zip(run_values, run_starts, run_lengths):
            plt.axvspan(s, s + l, color=palette[v % len(palette)], alpha=bg_alpha)

        # plot y_test markers
        if y_pred is not None:
            hop = len(self.y) // len(y_pred)
            zero = self.X.min()
            # markers = 'ovsP*+x1<p'

            for i, yi in enumerate(set(y_pred)):
                scale = 1 - i * 0.025 if zero > 0 else 1 + i * 0.025
                xs = np.argwhere(y_pred == yi).flatten() * hop + hop
                ys = np.ones(len(xs)) * zero * scale
                plt.scatter(xs, ys, marker='.', c=palette[i % len(palette)], s=2)

    def plot_class_distribution(self):
        """
        Plot histogram of classes' samples
        """
        fig, ax = plt.subplots()
        x = self.class_distribution.keys()
        height = self.class_distribution.values()
        bar = plt.bar(x, height)

        if len(self.class_labels):
            ax.set_xticks(np.arange(len(x)))
            ax.set_xticklabels(self.class_labels, rotation=70)

        return fig, ax, bar

    def _should_plot(self, force=False):
        """
        Test if plotting is enabled
        """
        if force:
            return True
        if self.__class__.is_plot_disabled:
            return False
        return True
