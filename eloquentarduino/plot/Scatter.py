import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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

    def show(self, **kwargs):
        """
        Show
        :param kwargs:
        :return:
        """
        ax = plt.figure().add_subplot()
        scatter = ax.scatter(self.X[:, 0].tolist(), self.X[:, 1].tolist(), c=self.hue, s=self.size, **kwargs)
        ax.legend(*scatter.legend_elements(), title="Classes")
        ax.set_xlabel('Component #1')
        ax.set_ylabel('Component #2')
        ax.set_title(self.title or '')
        plt.show()


def scatter(X, y=None, tsne=0, **kwargs):
    """
    Create scatter plot
    :param X:
    :param y:
    :param tsne:
    :return:
    """
    scatter = Scatter(X, y, **kwargs)

    if tsne > 0:
        scatter.tsne(tsne)

    scatter.show()