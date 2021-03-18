class GridSearchResult:
    """
    Result of a grid search iteration
    """
    def __init__(self, dataset, clf, accuracy, resources, inference_time):
        """
        
        """
        self.dataset = dataset
        self.clf = clf
        self.accuracy = accuracy
        self.resources = resources
        self.inference_time = inference_time