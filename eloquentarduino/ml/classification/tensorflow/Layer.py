from itertools import product
from tensorflow.keras import layers as tf_layers


class Layer:
    """
    Proxy for Tensorflow Keras layers
    """
    def __init__(self, layer_type, *args, **kwargs):
        self.type = layer_type
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        """
        Convert to string
        """
        return ('%s [%s, %s]' % (self.type, self.args, self.kwargs)).replace('(), ', '').replace(', {}', '')

    def __repr__(self):
        """
        Convert to string
        """
        return str(self)

    def __getstate__(self):
        """
        Get state for copy
        """
        return self.__dict__

    def __setstate__(self, state):
        """
        Set state from copy
        """
        self.__dict__ = state

    def __getattr__(self, layer_type):
        """
        Make the Layer proxy object to behave as a layer factory
        """
        assert hasattr(tf_layers, layer_type), 'unknown layer %s' % layer_type

        return Layer(layer_type)

    def __call__(self, *args, **kwargs):
        """
        Allow the proxy to be configured as if it was a "normal" keras layer
        """
        self.args = args
        self.kwargs = kwargs

        return self

    @property
    def tf_type(self):
        return getattr(tf_layers, self.type)

    def enumerate(self):
        """
        List all the combinations for the hyperparameters
        :return: generator of configured layers
        """
        fixed_parameters = {k: v for k, v in self.kwargs.items() if not k.startswith('hyper_')}
        hyper_parameters = {k.replace('hyper_', ''): v for k, v in self.kwargs.items() if k.startswith('hyper_')}

        if len(hyper_parameters) == 0:
            yield Layer(self.type, *self.args, **fixed_parameters)
            return

        for hyper_values in product(*hyper_parameters.values()):
            hyper_combination = {k: v for k, v in zip(hyper_parameters.keys(), hyper_values)}
            hyper_combination.update(fixed_parameters)

            yield Layer(self.type, *self.args, **hyper_combination)

    def instantiate(self):
        """
        Instantiate Keras layer
        """
        return self.tf_type(*self.args, **self.kwargs)


# factory-like instance
layers = Layer(None)