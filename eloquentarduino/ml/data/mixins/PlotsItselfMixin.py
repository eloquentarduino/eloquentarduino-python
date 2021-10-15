import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from umap import UMAP
from sklearn.model_selection import train_test_split


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

    def plot(self, title='', columns=None, n_ticks=15, grid=True, fontsize=6, bg_alpha=0.2, once_every=1, max_samples=None, palette=None, y_pred=None, force=False, linewidth=1, **kwargs):
        """
        Plot dataframe
        :param title: str title of plot
        :param columns: list columns to plot
        :param n_ticks: int number of ticks on the x axis
        :param grid: bool wether to display the grid
        :param fontsize: int font size for the axis values
        :param bg_alpha: float alpha of classes' background color
        :param once_every: int limit the number of samples to draw
        :param max_samples: int if set, limit the number of plotted samples (same as once_every=num_samples/max_samples)
        :param y_pred: np.array draw predictions markers on top of plot
        :param force: bool if True, always draw the plot, no matter disable_plot() calls
        :param linewidth: int line width for pandas.plot()
        """
        if not self._should_plot(force):
            print('Dataset plotting is disabled, skipping...')
            return

        plot_columns = [c for c in (columns or list(self.df.columns)) if c != 'y']
        idx = self._subsample_idx(self.y, max_samples)
        df = pd.DataFrame(self.df[plot_columns].iloc[idx].to_numpy(), columns=plot_columns)
        length = len(df)

        plt.figure()
        df.plot(title=title, xticks=range(0, length, length // n_ticks), grid=grid, fontsize=fontsize, rot=70, linewidth=linewidth, **kwargs)

        # highlight labels
        y = self.y[idx]
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
            hop = max(1, length / len(y_pred))
            zero = self.X.min()
            # markers = 'ovsP*+x1<p'

            if len(y_pred) > length:
                idx = self._subsample_idx(y_pred, length)
                y_pred = y_pred[idx]

            for i, yi in enumerate(set(y_pred)):
                scale = 1 - i * 0.025 if zero > 0 else 1 + i * 0.025
                xs = np.argwhere(y_pred == yi).flatten().astype(float) * hop + hop
                ys = np.ones(len(xs)) * zero * scale

                plt.scatter(xs, ys, marker='.', c=palette[i % len(palette)], s=2)

    def plot_each_class(self, **kwargs):
        """Plot each class on its own"""
        for yi in sorted(list(set(self.y))):
            mask = self.y == yi
            title = self.classmap.get(yi, "Class #%d" % yi)
            self.replace().mask(mask).plot(title=title, **kwargs)

    def plot_class_distribution(self, force=False):
        """
        Plot histogram of classes' samples
        :param force: bool if True, always draw the plot, no matter disable_plot() calls
        """
        if not self._should_plot(force):
            print('Dataset plotting is disabled, skipping...')
            return

        fig, ax = plt.subplots()
        x = self.class_distribution.keys()
        height = self.class_distribution.values()
        bar = plt.bar(x, height)

        if len(self.class_labels):
            ax.set_xticks(np.arange(len(x)))
            ax.set_xticklabels(self.class_labels, rotation=70)

        return fig, ax, bar

    def plot_class_durations(self, classes=None, bins=20, cumsum=False, force=False, **kwargs):
        """
        Plot freq histogram of each class durations
        :param classes: list list of classes to plot (default to None)
        :param bins: int number of bins for histogram (default to 20)
        :param cumsum: bool if True, plots cumsum of distribution (default to False)
        :param force: bool if True, always draw the plot, no matter disable_plot() calls
        :return: tuple (histogram, edges)
        """
        if not self._should_plot(force):
            print('Dataset plotting is disabled, skipping...')
            return

        y_segments = self.y_segments()

        for class_idx in set(self.y):
            if classes is not None and class_idx not in classes:
                continue

            durations = [duration for label, start, duration in y_segments if label == class_idx]
            class_name = self.classmap.get(class_idx, str(class_idx))

            if len(durations) == 0:
                continue

            plt.figure()
            plt.xlabel('Durations of class %s' % class_name)

            if cumsum:
                hist, xs = np.histogram(durations, bins=bins)
                plt.plot(xs[:-1], np.cumsum(hist))
                plt.ylabel('Cumsum(Count)')
            else:
                plt.hist(durations, bins=bins, **kwargs)
                plt.ylabel('Count')

            plt.ylim(ymin=0)

            hist, edges = np.histogram(durations, bins=bins)

            return np.cumsum(hist), edges[:-1]

    def plot_pairplot(self, max_samples=500, force=False, palette=None, **kwargs):
        """
        Draw pairplot of features, optionally applying feature rediction
        :param max_samples: int max number of points to plot
        :param force: bool if True, always draw the plot, no matter disable_plot() calls
        :param palette: list|dict|None palette for sns.pairplot()
        :param kwargs: dict passed to seaborn.pairplot()
        """
        if not self._should_plot(force):
            print('Dataset plotting is disabled, skipping...')
            return

        if self.length > max_samples:
            X, _0, y, _1 = train_test_split(self.X, self.y, train_size=max_samples)
            df = self.replace(X=X, y=y).df
        else:
            df = self.df

        if palette is None and self.num_classes < 10:
            palette = sns.color_palette("tab10")[:self.num_classes]

        sns.pairplot(df.astype({'y': 'int'}), hue='y', palette=palette, **kwargs)

    def dim_reduction(self, pca=0, tsne=0, umap=0, isomap=0, lle=0, lda=False, **kwargs):
        """
        Apply dimensionality reduction
        :param pca: int number of PCA components
        :param tsne: int number of tSNE components
        :param umap: int number of UMap components
        :param isomap: int number of Isomap components
        :param lle: int number of LocallyLinearEmbedding components
        :param lda: bool if True, apply LDA
        :param kwargs: dict arguments for the reducer
        :return: Dataset
        """
        assert pca + tsne + umap + isomap + lle + lda > 0, 'one of pca, tsne, umap, isomap, lda MUST be > 0'

        if pca > 0:
            n_components = pca
            reducer = PCA(n_components=n_components, svd_solver='randomized', **kwargs)
        elif tsne > 0:
            n_components = tsne
            reducer = TSNE(n_components=n_components, n_iter_without_progress=50, random_state=0)
        elif umap > 0:
            n_components = umap
            reducer = UMAP(n_components=n_components, **kwargs)
        elif umap > 0:
            n_components = umap
            reducer = UMAP(n_components=n_components, **kwargs)
        elif isomap > 0:
            n_components = isomap
            reducer = Isomap(n_components=n_components, **kwargs)
        elif lle > 0:
            n_components = lle
            reducer = LocallyLinearEmbedding(n_components=n_components, **kwargs)
        elif lda:
            n_components = min(self.num_classes - 1, self.num_features)
            reducer = LinearDiscriminantAnalysis(n_components=n_components, **kwargs)

        X = reducer.fit_transform(self.X, self.y)

        return self.replace(name='%s (%d dim)' % (self.name, n_components), X=X)

    def _should_plot(self, force=False):
        """
        Test if plotting is enabled
        """
        if force:
            return True
        if self.__class__.is_plot_disabled:
            return False
        return True

    def _subsample_idx(self, array, max_length):
        """
        Resize array to a given length at (almost) equally spaced intervals
        :param array: np.ndarray
        :param max_length: int
        :return: np.ndarray
        """
        max_length = max_length or 0

        if max_length > 0 and len(array) > max_length:
            idx = (np.arange(max_length, dtype=float) * len(array) / max_length)
            idx = np.unique(idx.astype(int))

            return idx

        return np.arange(len(array))