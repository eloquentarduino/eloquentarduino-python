class GridSearchResultCollection(list):
    """
    Add syntactic sugar to grid search result list
    """

    @property
    def best(self):
        """
        Get best pipeline
        """
        return self[0]

    @property
    def best_pipeline(self):
        """
        Get best pipeline
        """
        return self.best["pipeline"]

    def filter_by_score(self, min_score):
        """
        Remove items the have a score too low
        :param min_score: float
        :return: self
        """
        passes = []

        for i, result in enumerate(self):
            score = result["score"]

            if score >= min_score:
                passes.append(result)

        return GridSearchResultCollection(passes)

    def filter_by_missing_rate(self, max_missing_rate):
        """
        Remove items the have a missing rate too high
        :param max_missing_rate: float
        :return: self
        """
        passes = []

        for i, result in enumerate(self):
            if result.missing_rate < max_missing_rate:
                passes.append(result)

        return GridSearchResultCollection(passes)
