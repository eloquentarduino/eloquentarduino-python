import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.covariance import EllipticEnvelope


class Scatter:
    """
    Scatter plot
    """
    def __init__(self, X, y=None, hue=None, size=None, title=None):
        """

        :param X:
        :param y:
        :param hue:
        :param size:
        :param title: str
        """
        if len(X.shape) == 1 or X.shape[1] == 1:
            assert len(y) == len(X), 'when X is 1D, y MUST be a 1D array'
            X = np.hstack((X.reshape((-1, 1), np.asarray(y).reshape((-1, 1)))))

        self.X = X
        self.hue = hue
        self.size = size
        self.title = title
        self.patches = []

    def tsne(self, n_components=2, random_state=0, **kwargs):
        """
        Apply t-SNE
        :param n_components: int (default=2)
        :param random_state: int (default=0)
        :return: self
        """
        assert isinstance(n_components, int) and n_components >= 1, 'n_components MUST be positive'

        self.X = TSNE(n_components=n_components, random_state=random_state, **kwargs).fit_transform(self.X)

        return self

    def pca(self, n_components=2, random_state=0, **kwargs):
        """
        Apply PCA
        :param n_components: int (default=2)
        :param random_state: int (default=0)
        :return: self
        """
        assert isinstance(n_components, int) and n_components >= 1, 'n_components MUST be positive'

        self.X = PCA(n_components=n_components, random_state=random_state, **kwargs).fit_transform(self.X)

        return self

    def elliptic_envelope(self, **kwargs):
        """
        Drop outliers using EllipticEnvelope
        """
        X = self.X[:, :2]
        is_outlier = EllipticEnvelope(**kwargs).fit(X).predict(X) < 0
        self.X = X[~is_outlier]

        if self.hue is not None and hasattr(self.hue, 'shape'):
            self.hue = self.hue[~is_outlier]

        return self

    def add_patch(self, patch):
        """
        Add patch to plot
        """
        self.patches.append(patch)

    def show(self, **kwargs):
        """
        Show
        :param kwargs:
        :return: ax
        """
        ax = plt.figure().add_subplot()
        scatter = ax.scatter(self.X[:, 0], self.X[:, 1], c=self.hue, s=self.size, **kwargs)
        ax.legend(*scatter.legend_elements(), title="Classes")
        ax.set_xlabel('Component #1')
        ax.set_ylabel('Component #2')
        ax.set_title(self.title or '')

        for patch in self.patches:
            ax.add_patch(patch)

        plt.show()

        return ax


def scatter(X, y=None, tsne=0, pca=0, elliptic=False, patches=None, **kwargs):
    """
    Create scatter plot
    :param X:
    :param y:
    :param tsne:
    :param pca:
    :param elliptic:
    :param patches: list|None
    :return:
    """
    scatter = Scatter(X, y, **kwargs)

    if tsne > 0:
        scatter.tsne(tsne)

    if pca > 0:
        scatter.pca(pca)

    if elliptic:
        scatter.elliptic_envelope(**{k[10:]: v for k, v in kwargs.items() if k.startswith('elliptic_')})

    if isinstance(patches, list):
        for patch in patches:
            scatter.add_patch(patch)

    return scatter.show()