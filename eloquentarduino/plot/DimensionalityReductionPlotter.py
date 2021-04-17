import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DimensionalityReductionPlotter:
    """
    Plot PCA components
    """
    def __init__(self, X, y):
        """
        Constructor
        :param X:
        :param y:
        """
        self.X = X
        self.y = np.asarray(y)

    @property
    def dim(self):
        """
        Get X dimension
        """
        return self.X.shape[1]

    def sparse(self, sparsity):
        """
        Select only some of the samples
        :param sparsity: int
        :return: self
        """
        self.X = self.X[::sparsity]
        self.y = self.y[::sparsity]

        return self

    def pca(self, n=2):
        """
        Apply PCA
        :param n: int pca components
        :return: self
        """
        self.X = PCA(n_components=n).fit_transform(self.X)

        return self

    def tsne(self, n=2):
        """
        Applt t-SNE
        :param n: int t-SNE components
        :return: self
        """
        self.X = TSNE(n_components=n).fit_transform(self.X)

        return self

    def drop_outliers(self):
        """
        Drop outliers from data for more dense plotting
        :return: self
        """
        outliers = EllipticEnvelope().fit_predict(self.X, self.y)
        self.X = self.X[outliers > 0]
        self.y = self.y[outliers > 0]

        return self

    def plot(self):
        """
        Plot
        """
        assert self.dim in [2, 3], 'you MUST apply PCA/t-SNE to n=2 or n=3'

        if self.dim == 2:
            ax = plt.figure().add_subplot()
        else:
            ax = plt.figure().add_subplot(111, projection='3d')
            ax.set_zlabel('PCA component #3')

        scatter = ax.scatter(*self.X.T.tolist(), c=self.y)
        ax.legend(*scatter.legend_elements(), title="Classes")
        ax.set_xlabel('Component #1')
        ax.set_ylabel('Component #2')
        plt.show()
