class Dataset:
    """
    Abstraction of a dataset
    """
    def __init__(self, name, X, y):
        """
        :param name:
        :param X:
        :param y:
        """
        self.name = name
        self.X = X
        self.y = y