from copy import copy


class GridSearch:
    """
    Pipeline grid search
    """
    def __init__(self):
        self.paths = []

    @property
    def possibilities(self):
        """
        Better name for paths
        """
        return self.paths

    def then(self, steps):
        """
        Add more steps
        """
        self.paths = [path + copy(steps) for path in self.paths]

        return self

    def one_of(self, paths):
        """
        Add choice
        """
        new_paths = []

        for path in paths:
            if path is None:
                path = []
            if not isinstance(path, list):
                path = [path]

            new_paths = [copy(existing_path) + copy(path) for existing_path in self.paths]

        self.paths = new_paths

        return self

    def optionally_one_of(self, paths):
        """

        """
        return self.one_of([None] + paths)