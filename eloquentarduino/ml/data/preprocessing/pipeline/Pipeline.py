from eloquentarduino.ml.data import Dataset
from eloquentarduino.utils import jinja


class Pipeline:
    """
    Define a pre-processing pipeline that can be ported to plain C++
    """
    def __init__(self, name, dataset, steps):
        """
        Constructor
        :param name: str a name for the pipeline
        :param dataset: Dataset a dataset to train the pipeline on
        :param steps: list list of steps
        """
        assert isinstance(dataset, Dataset), 'dataset MUST be an instance of eloquentarduino.ml.data.Dataset'
        assert isinstance(steps, list), 'steps MUST be a list'

        self.name = name
        self.dataset = dataset
        self.steps = steps

        assert len([step.name for step in steps]) == len(set([step.name for step in steps])), 'steps names MUST be unique'

    def fit(self):
        """
        Fit the steps
        :return: self
        """
        for step in self.steps:
            self.dataset.X, self.dataset.y = step.fit(self.dataset.X, self.dataset.y)

        return self

    def transform(self, X):
        """
        Apply pipeline
        :param X:
        """
        #assert X.shape[1] == self.dataset.X.shape[1], 'X MUST match with the training X'

        for step in self.steps:
            X = step.transform(X)

        return X

    def port(self):
        """
        Port to C++
        """
        return jinja('ml/data/preprocessing/pipeline/Pipeline.jinja', {
            'name': self.name,
            'steps': self.steps
        }, pretty=True)


