from eloquentarduino.utils import jinja


class AIfESPort:
    """
    Port TensorFlow Neural Network to AIfES framework
    """
    def __init__(self, network, classname='NeuralNetwork', classmap=None):
        """
        :param network: NeuralNetwork
        :param classname: str
        :param classmap: dict
        """
        self.network = network
        self.classname = classname
        self.classmap = classmap

    def __str__(self):
        """
        Port
        """
        return jinja('ml/classification/tensorflow/aifes/NeuralNetwork.jinja', {
            'classname': self.classname,
            'num_inputs': self.network.num_inputs,
            'num_outputs': self.network.num_classes,
            'classmap': self.classmap
        })