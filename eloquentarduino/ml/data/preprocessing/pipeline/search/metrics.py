import numpy as np


def average_precision(y_true, y_pred):
    """
    Return average of precisions, no matter unbalanced classes
    :param y_true: numoy.ndarray
    :param y_pred: numoy.ndarray
    """
    scores = [((y_pred == y) & (y_pred == y_true)).sum() / (y_pred == y).sum() for y in set(y_true)]

    return sum(scores) / len(scores)


def average_recall(y_true, y_pred):
    """
    Return average of recalls, no matter unbalanced classes
    :param y_true: numoy.ndarray
    :param y_pred: numoy.ndarray
    """
    scores = [((y_true == y) & (y_pred == y_true)).sum() / (y_pred == y).sum() for y in set(y_true)]

    return sum(scores) / len(scores)


def class_precision(only=None, exclude=None):
    """
    Get average precision of given classes
    :param only: list only consider the given classes
    :param exclude: list exclude the given classes
    :return: float average precision score
    """
    def score(y_true, y_pred):
        nonlocal only
        nonlocal exclude
        if only is not None:
            if not isinstance(only, list):
                only = [only]

            mask = np.isin(y_pred, only)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        elif exclude is not None:
            if not isinstance(exclude, list):
                exclude = [exclude]

            mask = ~np.isin(y_pred, exclude)
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        return average_precision(y_true, y_pred)

    return score
