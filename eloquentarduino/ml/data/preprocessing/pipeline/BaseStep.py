from eloquentarduino.utils import jinja


class BaseStep:
    """
    Base class for the pipeline steps
    """
    def __init__(self, name):
        self.name = name
        self.input_dim = None
        self.inplace = False
        self.working_dim = 0
        self.includes = []

    def __str__(self):
        """
        Convert to string
        """
        config = self.get_config()
        config.update(input_dim=self.input_dim)
        config = ', '.join(['%s=%s' % (key, str(val)) for key, val in config.items()])

        return '%s (%s)' % (self.name, config)

    def __repr__(self):
        """
        Convert to string
        """
        return str(self)

    def get_config(self):
        """
        Get configuration options
        """
        return {}

    def set_X(self, X):
        """
        Update input dim
        :param X:
        """
        self.input_dim = X.shape[1]

    def fit(self, X, y):
        """
    Fit step to input
        """
        raise NotImplemented

    def transform(self, X, y=None):
        """
        Transform input
        :param X:
        :param y:
        """
        raise NotImplemented

    def get_template_data(self):
        """
        Get data for jinja template
        :return: dict
        """
        raise NotImplemented('get_template_data')

    def include_c_library(self, library):
        """
        Include library headers once ported to C++
        """
        self.includes.append(library)

    def port(self, ns):
        """
        Port to plain C++
        :param ns: str namespace for the pipeline
        :return: str C++ code
        """
        template_name = type(self).__name__
        template_data = self.get_template_data()
        template_data.update(name=self.name, input_dim=self.input_dim, working_dim=self.working_dim, ns=ns)

        return self.postprocess_port(jinja('ml/data/preprocessing/pipeline/%s.jinja' % template_name, template_data))

    def postprocess_port(self, ported):
        """
        Apply post-processing to ported code
        :param ported: str
        :return: str
        """
        return ported
