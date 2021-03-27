import numpy as np
from collections import Counter
from eloquentarduino.plot.ConfusionMatrix import ConfusionMatrix


class RankMatrix:
    """
    Plot a matrix where each row shows the positions of an item
    """
    def __init__(self, items):
        """
        :param items: dict
        """
        assert isinstance(items, dict), 'items MUST be a dict'
        for name, ranks in items.items():
            assert isinstance(name, str), 'keys MUST be strings'
            assert isinstance(ranks, list), 'values MUST be lists'

        # delegate to a confusion matrix the drawing
        m = min([min(ranks) for ranks in items.values()])
        M = max([max(ranks) for ranks in items.values()])

        self.y_true = []
        self.y_pred = []
        self.labels = list(items.keys())

        for row, (name, ranks) in enumerate(items.items()):
            counts = Counter(ranks)
            for i in range(m, M + 1):
                if i in counts:
                    self.y_true += [row + 1] * counts[i]
                    self.y_pred += [i] * counts[i]

    def show(self, **kwargs):
        """
        Show matrix
        """
        ConfusionMatrix(self.y_true, self.y_pred, labels=self.labels).show(**kwargs)

