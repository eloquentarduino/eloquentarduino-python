import numpy as np
from scipy.stats import mode
from eloquentarduino.ml.data.pipeline.BaseStep import BaseStep
from eloquentarduino.ml.data.preprocessing import RollingWindow as Window


class RollingWindow(BaseStep):
    """
    Process data as a rolling window
    """
    def __init__(self, depth, shift=1, flatten=True):
        """
        :param depth: int how many samples will form a single window
        :param shift: int how many samples to skip between windows
        :param flatten: bool if rolling window output should be flattened
        """
        self.window = Window(depth=depth, shift=shift)
        self.flatten = flatten

    def transform(self, X, y):
        """
        Transform data
        :param X: inputs
        :param y: labels
        :return: np.ndarray transformed input
        """
        windows = self.window.transform(np.hstack((X, y.reshape((-1, 1)))), flatten=False)
        X = [window[:, :-1] for window in windows]
        y = [mode(window[:, -1])[0][0] for window in windows]

        if self.flatten:
            X = [x.flatten() for x in X]

        return X, y

    def port(self, class_name='RollingWindow'):
        """
        Port to C++
        :return: str
        """
        assert self.input_dim is not None, 'Unfitted'
        return jinja('preprocessing/RollingWindow.jinja', {
            'class_name': class_name,
            'depth': self.depth,
            'input_dim': self.input_dim
        })
