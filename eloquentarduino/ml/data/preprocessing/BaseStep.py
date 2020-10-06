from eloquentarduino.utils import jinja


class BaseStep:
    """Abstract class for pre-processing steps"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.input_shape = X.shape
        self.output_shape = None

    @property
    def name(self):
        """Get name for step"""
        raise NotImplementedError("%s MUST implement name getter" % __class__)

    @property
    def input_dim(self):
        """Get input features count"""
        return self.input_shape[1]

    @property
    def output_dim(self):
        """Get output features count"""
        return self.output_shape[1]

    @property
    def inplace(self):
        """Get wether this step overrides input while working"""
        return False

    def __str__(self):
        self.not_implemented('__str__')

    def not_implemented(self, function):
        """Raise NotImplementedError"""
        raise NotImplementedError('%s MUST implement %s' % (self.__class__, function))

    def apply(self, Xt):
        """Apply transformation"""
        self.output_shape = Xt.shape
        return Xt

    def jinja(self, template, data={}):
        """Return Jinja2 template"""
        data.update(input_dim=self.input_dim, output_dim=self.output_dim)
        return jinja('Pipeline/%s' % template, data)

    def transform(self):
        """Apply transformation"""
        self.not_implemented('transform')

    def port(self):
        """Port to plain C"""
        raise self.not_implemented('port')

    def describe(self, *args):
        params_description = ', '.join([self.describe_param(*param) for param in args])
        dim_description = 'input_dim=%d, output_dim=%d' % (self.input_dim, self.output_dim)
        return '%s(%s, %s)' % (self.name, params_description, dim_description)

    def describe_param(self, label, value):
        value = ('True' if value else 'False') if isinstance(value, bool) else str(value)
        return '%s=%s' % (label, value)