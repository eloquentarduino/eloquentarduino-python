import numpy as np


def average_precision(y_true, y_pred):
    """
    Return average of precisions, no matter unbalanced classes
    :param y_true: numoy.ndarray
    :param y_pred: numoy.ndarray
    """
    scores = [((y_pred == y) & (y_pred == y_true)).sum() / (y_pred == y).sum() for y in set(y_true)]
    scores = [score for score in scores if not np.isnan(score)]

    return sum(scores) / len(scores) if len(scores) > 0 else 0


def average_recall(y_true, y_pred):
    """
    Return average of recalls, no matter unbalanced classes
    :param y_true: numoy.ndarray
    :param y_pred: numoy.ndarray
    """
    scores = [((y_true == y) & (y_pred == y_true)).sum() / (y_pred == y).sum() for y in set(y_true)]
    scores = [score for score in scores if not np.isnan(score)]

    return sum(scores) / len(scores) if len(scores) > 0 else 0


def class_precision(only=None, exclude=None):
    """
    Get average precision of given classes
    :param only: list only consider the given classes
    :param exclude: list exclude the given classes
    :return: float average precision score
    """
    return class_score(average_precision)(only=only, exclude=exclude)


def class_recall(only=None, exclude=None):
    """
    Get average recall of given classes
    :param only: list only consider the given classes
    :param exclude: list exclude the given classes
    :return: float average precision score
    """
    return class_score(average_recall)(only=only, exclude=exclude)


def class_score(metric):
    """
    Generator for class scores (precision, recall)
    """
    def scorer(only=None, exclude=None):
        def score(y_true, y_pred):
            nonlocal only
            nonlocal exclude

            if only is not None:
                if not isinstance(only, list):
                    only = [only]

                try:
                    mask = np.isin(y_pred, only)
                    y_true = y_true[mask]
                    y_pred = y_pred[mask]
                except IndexError:
                    return 0
            elif exclude is not None:
                if not isinstance(exclude, list):
                    exclude = [exclude]

                try:
                    mask = ~np.isin(y_pred, exclude)
                    y_true = y_true[mask]
                    y_pred = y_pred[mask]
                except IndexError:
                    return 0

            return metric(y_true, y_pred)

        return score

    return scorer



def list_metrics():
    """
    List available metrics
    """
    return [
        'average_precision',
        'average_recall',
        'class_precision',
        'class_recall'
    ]
