from micromlgen import port


class SklearnClassifier:
    """
    Abstract base class for classifiers from the sklearn package
    """
    def reset(self):
        """
        Reset the classifier
        """
        pass

    def port(self, classname=None, classmap=None, **kwargs):
        """
        Port to plain C++
        :param classname: str name of the ported class
        :param classmap: dict classmap in the format {class_idx: class_name}
        """
        return port(self, classname=classname, classmap=classmap, **kwargs)