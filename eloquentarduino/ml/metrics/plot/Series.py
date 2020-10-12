class Series:
    """
    Abstract series manipulation for plots
    """
    def __init__(self, *args, **kwargs):
        """
        Create a series from a list of datasets
        (at least xs and ys is assumed)
        :param args: list of series (all of equal length)
        :param kwargs:
        """
        self._series = list(args)
        self._kwargs = kwargs
        self._filter = lambda *args: True
        self._map = lambda x, y, *args: y

    @property
    def xs(self):
        """
        Get x list
        :return:
        """
        return [x for x, *args in self.filtered]

    @property
    def filtered(self):
        """
        Get only items that passes filter
        :return:
        """
        return [values for values in zip(*self._series) if self._filter(*values)]

    @property
    def ys(self):
        """
        Get y list
        :return:
        """
        return [self._map(*values) for values in self.filtered]

    def plot(self, *args, **kwargs):
        """
        Abstract method to actually plot the series
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError('%s MUST implement plot()', __class__)

    def series(self, idx):
        """
        Get series after filter and map
        :param idx: index of the series
        :return: list
        """
        assert 0 < idx < len(self._series), 'Index %d is out of range [1, %d]' % (idx, len(self._series) - 1)
        return [self._map(values[0], values[1:]) for values in zip(*([self._series[idx]] + self._series)) if self._filter(*values)]

    def filter(self, f):
        """
        Filter data
        :param f: callable
        :return:
        """
        assert callable(f), 'filter MUST be callable'
        self._filter = f
        return self

    def map(self, f):
        """
        Map data
        :param f: callable
        :return:
        """
        assert callable(f), 'map MUST be callable'
        self._map = f
        return self