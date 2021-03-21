from micromlgen import port
from eloquentarduino.ml.classification.device import ClassifierResources


class SklearnClassifier:
    """
    Abstract base class for classifiers from the sklearn package
    """
    @property
    def sklearn_base(self):
        return [base for base in self.__class__.__bases__ if base.__module__.startswith('sklearn.')][0]

    def fit(self, X, y):
        """
        Fit
        """
        self.sklearn_base.fit(self, X, y)
        # keep track of X and y
        self.X = X
        self.y = y

        return self

    def reset(self):
        """
        Reset the classifier
        """
        pass

    def hyperparameters_grid(self, X=None):
        """
        Get default hyperparameters values from grid search
        :param X:
        """
        return {}

    def port(self, classname=None, classmap=None, **kwargs):
        """
        Port to plain C++
        :param classname: str name of the ported class
        :param classmap: dict classmap in the format {class_idx: class_name}
        """
        return port(self, classname=classname, classmap=classmap, **kwargs)

    def on_device(self, project=None):
        """
        Get device benchmarker
        :param project: Project
        """
        return ClassifierResources(self, project=project)