from eloquentarduino.ml.classification.abstract.GridSearchResult import GridSearchResult as Base


class GridSearchResult(Base):
    """

    """
    def __init__(self, hyperparameters, **kwargs):
        super().__init__(**kwargs)
        self.hyperparameters = hyperparameters