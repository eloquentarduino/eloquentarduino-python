import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import quantile_transform
from sklearn.covariance import EllipticEnvelope


class PCAPlotter:
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

    def plot(self, n=2, outliers=True, sparsity=0):
        """
        Plot
        :param n: int either 2 or 3
        :param outliers: bool if to remove outliers
        :param sparsity: int how many samples to skip
        """
        assert n == 2 or n == 3, 'n MUST be 2 or 3'
        assert sparsity >= 0, 'sparsity MUST be non-negative'

        X = self.X[::(1 + sparsity)]
        y = self.y[::(1 + sparsity)]

        pca = PCA(n_components=n).fit_transform(X)
        y = y

        if n == 2:
            ax = plt.figure().add_subplot()
        else:
            ax = plt.figure().add_subplot(111, projection='3d')
            ax.set_zlabel('PCA component #3')

        if not outliers:
            outliers = EllipticEnvelope().fit_predict(pca, None)
            pca = pca[outliers > 0]
            y = y[outliers > 0]

        scatter = ax.scatter(*pca.T.tolist(), c=y)
        ax.legend(*scatter.legend_elements(), title="Classes")
        ax.set_xlabel('PCA component #1')
        ax.set_ylabel('PCA component #2')

        if n == 3:
            ax.set_zlabel('PCA component #3')

        plt.show()
