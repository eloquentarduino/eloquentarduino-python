import matplotlib.pyplot as plt


class Bar:
    """
    A bar plot where you can append data dynamically
    """
    def __init__(self, xs=None, ys=None, labels=None):
        self.xs = xs or []
        self.ys = ys or []
        self.labels = labels or []
        self.options = {
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

    def autolimit(self):
        """
        Set y limit based on data
        """
        m = min(self.ys) * 0.8
        M = max(self.ys) * 1.2
        self.ylim(m, M)

    def sort(self):
        """
        Sort by y
        """
        self.labels = [l for _, l in sorted(zip(self.ys, self.labels))]
        self.ys = sorted(self.ys)

    def show(self, title=None, legend=False):
        fig, ax = plt.subplots()

        for x, y, label in zip(self.xs, self.ys, self.labels):
            ax.bar(x, y, label=label)

        if legend:
            ax.legend()

        if self.options['ylim']:
            ax.set_ylim(*self.options['ylim'])

        if title:
            ax.set_title(title)

        plt.show()