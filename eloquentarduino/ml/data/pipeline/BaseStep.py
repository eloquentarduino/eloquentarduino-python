import numpy as np
from eloquentarduino.utils import jinja


class BaseStep:
    """
    Abstract class for pre-processing steps
    """
    @property
    def name(self):
        """
        Get name for step
        """
        return type(self).__name__

    @property
    def num_inputs(self):
        """
        Get number of inputs
        """
        return self.input_shape[1]

    @property
    def num_outputs(self):
        """
        Get number of outputs
        """
        return self.output_shape[1]

    @property
    def inplace(self):
        """
        Get wether this step overrides input while working
        """
        return False

    def __str__(self):
        """
        Convert to string
        """
        self.not_implemented('__str__')

    def not_implemented(self, function):
        """
        Raise NotImplementedError
        """
        raise NotImplementedError('%s MUST implement %s' % (self.__class__, function))

    def apply(self, X, y):
        """
        Apply step transformation
        """
        Xt, yt = self.transform(X, y)
        Xt = np.asarray(Xt)
        yt = np.asarray(yt)

        self.input_shape = X.shape
        self.output_shape = Xt.shape

        return Xt, yt

    def jinja(self, template, data={}):
        """
        Return rendered template
        :param template: str template name
        :param data: dict template data
        """
        data.update(input_dim=self.input_dim, output_dim=self.output_dim)

        return jinja('Pipeline/%s' % template, data)

    def transform(self, X, y):
        """
        Transform X and y
        """
        self.not_implemented('transform')