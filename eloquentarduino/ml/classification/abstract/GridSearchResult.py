class GridSearchResult:
    """
    Result of a grid search iteration
    """
    def __init__(self, dataset, clf, accuracy, resources=None, inference_time=None, passes=False, fail_reason=None):
        """
        
        """
        self.dataset = dataset
        self.clf = clf
        self.accuracy = accuracy
        self.resources = resources
        self.inference_time = inference_time
        self.passes = passes
        self.fail_reason = fail_reason
