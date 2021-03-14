import matplotlib.pyplot as plt


class Bar:
    """
    A bar plot where you can append data dynamically
    """
    def __init__(self):
        self.xs = []
        self.ys = []
        self.labels = []
        self.options = {
            'legend': False,
            'ylim': None
        }

    def append(self, y, x=None, label=None):
        """
        Append new bar
        :param y: float
        :param x: int
        :param label: str
        """
        self.xs.append(x or len(self.xs))
        self.ys.append(y)
        self.labels.append(label)

    def legend(self, legend=True):
        """
        Toggle legend
        :param legend: bool
        """
        self.options['legend'] = legend

    def ylim(self, m, M):
        """
        Set y limit
        """
        self.options['ylim'] = (m, M)

    def show(self):
        fig, ax = plt.subplots()

        for x, y, label in zip(self.xs, self.ys, self.labels):
            ax.bar(x, y, label=label)

        if self.options['legend']:
            ax.legend()

        if self.options['ylim']:
            ax.set_ylim(*self.options['ylim'])

        plt.show()