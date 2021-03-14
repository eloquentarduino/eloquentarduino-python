import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
        self.y = y

    def plot(self, n=2):
        """
        Plot
        :param n: int either 2 or 3
        """
        assert n == 2 or n == 3, 'n MUST be 2 or 3'

        if n == 2:
            ax = plt.figure().add_subplot()
        else:
            ax = plt.figure().add_subplot(111, projection='3d')

        pca = PCA(n_components=n).fit_transform(self.X)
        scatter = ax.scatter(pca[:, 0], pca[:, 1], c=self.y)

        ax.legend(*scatter.legend_elements(), title="Classes")
        ax.set_xlabel('PCA component #1')
        ax.set_ylabel('PCA component #2')

        if n == 3:
            ax.set_zlabel('PCA component #3')

        plt.show()
