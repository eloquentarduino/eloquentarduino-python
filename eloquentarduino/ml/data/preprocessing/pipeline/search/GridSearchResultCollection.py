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

    def print_all(self, n=5):
        """
        Print results
        :param n: int
        """
        for result in self[:n]:
            print(result["pipeline"])
            print("\n--- Scores ---\n")
            result.print_scores()
            print("\n===============================\n")

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

    def sort_by_accuracy(self, only_classes=None):
        """
        Sort results by accuracy on given classes
        :param only_classes: list or None
        :return: GridSearchResultCollection
        @added 0.1.19
        """
        if only_classes is None:
            return GridSearchResultCollection(
                sorted(self, key=lambda result: result.accuracy_score, reverse=True)
            )
        else:
            return GridSearchResultCollection(
                sorted(self, key=lambda result: result.get_accuracy_score_of_classes(only_classes), reverse=True)
            )

    def sort_by_precision(self, only_classes=None):
        """
        Sort results by precision on given classes
        :param only_classes: list or None
        :return: GridSearchResultCollection
        @added 0.1.19
        """
        if only_classes is None:
            return GridSearchResultCollection(
                sorted(self, key=lambda result: result.precision_score, reverse=True)
            )
        else:
            return GridSearchResultCollection(
                sorted(self, key=lambda result: result.get_precision_score_of_classes(only_classes), reverse=True)
            )

    def sort_by_recall(self, only_classes=None):
        """
        Sort results by recall on given classes
        :param only_classes: list or None
        :return: GridSearchResultCollection
        @added 0.1.19
        """
        if only_classes is None:
            return GridSearchResultCollection(
                sorted(self, key=lambda result: result.recall_score, reverse=True)
            )
        else:
            return GridSearchResultCollection(
                sorted(self, key=lambda result: result.get_recall_score_of_classes(only_classes), reverse=True)
            )

    def sort_by_f1(self, only_classes=None):
        """
        Sort results by F1 on given classes
        :param only_classes: list or None
        :return: GridSearchResultCollection
        @added 0.1.19
        """
        if only_classes is None:
            return GridSearchResultCollection(
                sorted(self, key=lambda result: result.f1_score, reverse=True)
            )
        else:
            return GridSearchResultCollection(
                sorted(self, key=lambda result: result.get_f1_score_of_classes(only_classes), reverse=True)
            )
