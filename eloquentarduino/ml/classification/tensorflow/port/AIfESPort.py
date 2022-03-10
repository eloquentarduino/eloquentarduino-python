from eloquentarduino.utils import jinja
from tensorflow.keras.layers import Dense


class AIfESPort:
    """
    Port TensorFlow Neural Network to AIfES framework
    """
    def __init__(self, network, trainable=False, weights_decay=0, classname='NeuralNetwork', classmap=None, **kwargs):
        """
        :param network: NeuralNetwork
        :param trainable: bool If True, the ported network can be trained online (default to False)
        :param classname: str
        :param classmap: dict
        """
        self.network = network
        self.trainable = trainable
        self.weights_decay = weights_decay
        self.classname = classname
        self.classmap = classmap

    def __str__(self):
        """
        Port
        """
        for i, layer in enumerate(self.network.sequential.layers):
            assert isinstance(layer, Dense), 'All layers MUST be Dense. Layer %d is not (it\'s %s)' % (i, str(type(layer)))

        layers = [{
            'units': d.units,
            'weights': d.get_weights()[0].flatten(),
            'offline_weights': d.get_weights()[0].flatten() * (1 - self.weights_decay),
            'bias': d.get_weights()[1],
            'offline_bias': d.get_weights()[1] * (1 - self.weights_decay),
        } for d in self.network.sequential.layers]

        activations = [{
            'type': self.get_activation_type(d, i)
        } for i, d in enumerate(self.network.sequential.layers)]

        return jinja('ml/classification/tensorflow/aifes/NeuralNetwork.jinja', {
            'classname': self.classname,
            'num_inputs': self.network.num_inputs,
            'num_outputs': self.network.num_classes,
            'classmap': self.classmap,
            'layers': layers,
            'activations': activations,
            'trainable': self.trainable,
            'merge_weights': self.weights_decay > 0,
            'weights_decay': self.weights_decay
        })

    def get_activation_type(self, dense, index=None):
        """
        Get activation type for given dense layer
        :param dense: Layer
        :return: str
        """
        # @todo is there a more reliable way to perform this?
        activation = str(dense.activation).split(' ')[1]

        assert activation in ['relu', 'sigmoid', 'softmax'], 'unknown activation "%s" at index %s' % (activation, str(index))

        return activation
