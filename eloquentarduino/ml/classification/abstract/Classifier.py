from sklearn.model_selection import KFold
from eloquentarduino.ml.data import Dataset


class Classifier:
    def cross_val_score(self, dataset, num_folds=3):
        """
        Compute cross validation accuracy
        :param dataset: Dataset
        :param num_folds: int
        :return: float cross validation score
        """
        assert isinstance(dataset, Dataset), 'dataset MUST be an instance of Dataset'
        assert num_folds > 1, 'num_fold MUST be greather than 1'

        kfold = KFold(n_splits=num_folds, shuffle=True)
        scores = []

        for train_idx, test_idx in kfold.split(dataset.X, dataset.y):
            clf = self.clone()
            X_train = dataset.X[train_idx]
            y_train = dataset.y[train_idx]
            X_test = dataset.X[test_idx]
            y_test = dataset.y[test_idx]

            clf.fit(X_train, y_train)
            scores.append(clf.score(X_test, y_test))

        return sum(scores) / len(scores)