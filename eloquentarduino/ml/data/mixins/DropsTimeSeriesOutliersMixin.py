import numpy as np
from eloquentarduino.ml.data.mixins.dtwnumba import dtw_matrix


class DropsTimeSeriesOutliersMixin:
    """Implement time-series anomaly detection algorithms

    """
    def mark_outliers_by_dtw(self, window_size, max_distance_q=0.98, min_columns=1, outlier_class=None, outlier_class_name='Outliers'):
        """
        Mark chunks of data with a unusually high DTW distance
        :param window_size: int
        :param max_distance_q: float in range [0, 1]
        :param min_columns: int

        Returns
        -------
            A new Dataset without outliers
        """
        from eloquentarduino.ml.data.preprocessing.pipeline.Window import Window

        assert min_columns >= 1, 'min_columns MUST be greater than 0'

        outliers = np.zeros(len(self.X), dtype=bool)
        offset = 0

        for yi, Xi in self.iterate_classes():
            y_mock = np.zeros(len(Xi))
            class_outliers = {}

            for feature_idx in range(Xi.shape[1]):
                Xif = Xi[:, feature_idx].reshape((-1, 1))
                X, _ = Window(length=window_size, shift=window_size//2).fit(Xif, y_mock)
                dmatrix = dtw_matrix(X)
                min_distances = [row[row > 0].min() for row in dmatrix]
                max_min_distance = np.quantile(min_distances, q=max_distance_q)

                for window_idx, distances in enumerate(dmatrix):
                    min_distance = distances[distances > 0].min()

                    if min_distance > max_min_distance:
                        class_outliers.setdefault(window_idx, 0)
                        class_outliers[window_idx] += 1

            for window_idx, count in class_outliers.items():
                if count >= min_columns:
                    start = offset + window_idx * window_size // 2
                    outliers[start:start + window_size] = 1

            offset += len(Xi)

        # update labels and classmap
        if outlier_class is None:
            outlier_class = self.next_class

        new_y = np.where(outliers, outlier_class, self.y)
        new_classmap = {**self.classmap, **{outlier_class: outlier_class_name}}
        new_dataset = self.replace(y=new_y, classmap=new_classmap)
        new_dataset.outlier_class = outlier_class

        return new_dataset

    def drop_outliers_by_dtw(self, window_size, max_distance_q=0.98, min_columns=1, **kwargs):
        """
        Drop chunks of data with a unusually high DTW distance
        :param window_size: int
        :param max_distance_q: float in range [0, 1]
        :param min_columns: int

        Returns
        -------
            If return_indices is True, the indices marked as outliers
            Else, a new Dataset without outliers
        """
        outlier_class = self.next_class

        return (self
                .mark_outliers_by_dtw(
                    window_size=window_size,
                    max_distance_q=max_distance_q,
                    min_columns=min_columns,
                    outlier_class=outlier_class)
                .drop_class(outlier_class))

    # def drop_singular_spectrum(self, window_size, columns=None, **kwargs):
    #     """
    #     :param window_size: int
    #     :param columns: list or None
    #     """
    #     is_outlier = np.zeros(len(self.X))
    #
    #     for i in range(self.num_features):
    #         if columns is not None and i not in columns:
    #             continue
    #
    #         ts = self.X[:, i]
    #         model = SST(w=window_size, **kwargs)
    #         scores = model.detect(ts)
    #         is_outlier = np.logical_or(is_outlier, scores > scores.mean())
    #
    #     return self.mask(~is_outlier)
