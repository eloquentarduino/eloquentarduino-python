import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class KMeansOutlier(BaseStep):
    """
    Outlier detection algorithm based on k-means clustering
    and euclidean distance
    """
    def __init__(self, k=5, percentile=95, tol=0, add_feature=False, predict=True, name='KMeansOutlier', random_state=0):
        """
        :param k: int centroids for each class
        :param percentile: int (default=95)
        :param tol: float (default=0)
        :param add_feature: bool (default=False) if True, a new column is added to the inputs with 0 if inlier, 1 if outlier
        :param predict: bool (default=True) if True, returns 0 for inliers, 1 for outliers
        """
        assert k >= 1, 'k MUST be a positive number'

        super().__init__(name=name)
        self.k = k
        self.percentile = percentile
        self.tol = tol
        self.add_feature = add_feature
        self.predict = predict
        self.random_state = random_state
        self.centroids = None
        self.thresholds = None

    def fit(self, X, y):
        """

        """
        self.set_X(X)

        if self.add_feature:
            self.working_dim = self.input_dim + 1

        y_unique = np.sort(np.unique(y))
        centroids = []
        thresholds = []

        for class_idx in y_unique:
            X_class = X[y == class_idx]
            kmeans = KMeans(n_clusters=self.k).fit(X_class)
            centroids += kmeans.cluster_centers_.tolist()

            for centroid in kmeans.cluster_centers_:
                distances = euclidean_distances(X_class, [centroid])
                distance = (1 + self.tol) * np.percentile(distances, self.percentile)
                thresholds.append(distance)

        self.centroids = np.asarray(centroids)
        self.thresholds = np.asarray(thresholds)

        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        """
        assert self.centroids is not None, 'Unfitted'

        y_pred = np.zeros(len(X), dtype=np.bool)

        for centroid, threshold in zip(self.centroids, self.thresholds):
            distances = euclidean_distances(X, [centroid]).min(axis=1)
            y_pred = y_pred | (distances <= threshold)

        is_outlier = 1 - y_pred.astype(np.uint8)

        if self.add_feature:
            return np.hstack((X, is_outlier.reshape((-1, 1)))), y

        if self.predict:
            return X, is_outlier

        return X[~is_outlier], y[~is_outlier] if y is not None else None

    def get_template_data(self):
        """

        """
        return {
            'centroids': self.centroids,
            'thresholds': self.thresholds,
            'add_feature': self.add_feature,
            'predict': self.predict
        }