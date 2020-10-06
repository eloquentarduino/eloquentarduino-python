import matplotlib.pyplot as plt


def scatter_plot(X, y=None, axes=None, show=True, **kwargs):
    """Draw scatter plot of 2 features"""
    assert axes is not None, "axes MUST be a tuple"
    ax1, ax2 = axes
    plt.scatter(X[:, ax1], X[:, ax2], c=y)
    if show:
        plt.show()