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

    @property
    def X(self):
        """
        Get X
        """
        return self.dataset.X

    @property
    def input_dim(self):
        """
        Get input dim of the whole pipeline
        """
        return self.steps[0].input_dim

    @property
    def working_dim(self):
        """
        Get work data size
        """
        return max([step.working_dim for step in self.steps])

    @property
    def output_dim(self):
        """
        Get output size (works only after fitting)
        """
        return self.dataset.X.shape[1]

    @property
    def includes(self):
        """
        Get list of included libraries
        """
        return [library for step in self.steps for library in step.includes]

    def fit(self):
        """
        Fit the steps
        :return: self
        """
        for step in self.steps:
            self.dataset = self.dataset.replace(*step.fit(self.dataset.X, self.dataset.y))

        return self

    def transform(self, X):
        """
        Apply pipeline
        :param X:
        """

        for step in self.steps:
            X = step.transform(X)

        return X

    def port(self):
        """
        Port to C++
        """
        return jinja('ml/data/preprocessing/pipeline/Pipeline.jinja', {
            'name': self.name,
            'pipeline': self.name,
            'steps': self.steps,
            'input_dim': self.input_dim,
            'working_dim': max([1, self.working_dim, self.output_dim]),
            'includes': self.includes
        }, pretty=True)


