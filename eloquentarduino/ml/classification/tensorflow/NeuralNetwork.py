import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tinymlgen import port
from functools import reduce
from eloquentarduino.utils import jinja


Layer = namedtuple('Layer', 'layer args kwargs')


class NeuralNetwork:
    """
    Tensorflow neural network abstraction
    """
    def __init__(self):
        self.reset()

    @property
    def num_inputs(self):
        return reduce(lambda x, prod: x * prod, self.X.shape[1:], 1)

    @property
    def num_classes(self):
        return self.y.shape[1]

    def reset(self):
        """
        Reset the network
        """
        self.layer_definitions = []
        self.layers = []
        self.sequential = None
        self.history = None
        self.X = None
        self.y = None
        self.options = {
            'epochs': 20,
            'valid_size': 0.2,
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy'],
            'optimizer': 'rmsprop',
            'batch_size': 8
        }

    def add_dense(self, *args, **kwargs):
        """
        Add dense layer
        """
        return self.add_layer(layers.Dense, *args, **kwargs)

    def add_softmax(self):
        """
        Add softmax layer
        """
        return self.add_dense(units='num_classes', activation='softmax')

    def add_conv2d(self, *args, **kwargs):
        """
        Add Conv2D layer
        """
        return self.add_layer(layers.Conv2D, *args, **kwargs)

    def add_flatten(self):
        """
        Add flatten layer
        """
        return self.add_layer(layers.Flatten)

    def add_layer(self, layer, *args, **kwargs):
        """
        Add generic layer
        :param layer: layers.Layer
        """
        assert str(layer.__module__).startswith('tensorflow.python.keras.layers'), 'layer MUST be a tensorflow.keras layer'

        self.layer_definitions.append(Layer(layer=layer, args=args, kwargs=kwargs))

        return self

    def validate_on(self, percent):
        """
        Set validation percent size
        :param percent: float
        """
        self.options['valid_size'] = percent

    def set_epochs(self, epochs):
        """
        Set number of epochs for training
        :param epochs: int
        """
        self.options['epochs'] = epochs

    def set_optimizer(self, optimizer):
        """
        Set optimizer
        :param optimizer: str
        """
        self.options['optimizer'] = optimizer

    def set_loss(self, loss):
        """
        Set loss function
        :param loss: str
        """
        self.options['loss'] = loss

    def set_metrics(self, metrics):
        """
        Set metrics
        :param metrics: list
        """
        self.options['metrics'] = metrics

    def set_batch_size(self, batch_size):
        """
        Set batch_size
        :param batch_size: int
        """
        self.options['batch_size'] = batch_size

    def fit(self, X, y):
        """
        Fit the network
        :param X:
        :param y:
        """
        self.layers = []
        self.sequential = tf.keras.Sequential()
        y = self.to_categorical(y)

        # build the network
        for i, layer_definition in enumerate(self.layer_definitions):
            kwargs = layer_definition.kwargs

            if kwargs.get('units', 0) == 'num_classes':
                kwargs['units'] = y.shape[1]

            if i == 0 and 'input_shape' not in kwargs:
                kwargs['input_shape'] = X.shape[1:]

            layer = layer_definition.layer(*layer_definition.args, **kwargs)

            self.layers.append(layer)
            self.sequential.add(layer)

        validation_data = None

        if self.options['valid_size'] > 0:
            X, X_valid, y, y_valid = train_test_split(X, y, test_size=self.options['valid_size'])
            validation_data = (X_valid, y_valid)

        self.sequential.compile(optimizer=self.options['optimizer'], loss=self.options['loss'], metrics=self.options['metrics'])
        self.history = self.sequential.fit(X, y, epochs=self.options['epochs'], validation_data=validation_data, batch_size=self.options['batch_size'])
        self.X = X
        self.y = y

    def score(self, X, y):
        """
        Compute score on given data
        :param X:
        :param y:
        :return: float accuracy score
        """
        y = self.to_categorical(y)

        return self.sequential.evaluate(X, y)[1]

    def summary(self, *args, **kwargs):
        """
        Get topology summary
        """
        return self.sequential.summary(*args, **kwargs)

    def plot_train_loss(self, skip=2):
        """
        Plot train loss
        :param skip: int how many steps to skip at the beginning of the plot
        """
        plt.title('Loss')
        plt.plot(self.history.history['loss'][skip:], label='train')
        plt.plot(self.history.history['val_loss'][skip:], label='validation')
        plt.legend()
        plt.show()

    def plot_train_accuracy(self, skip=2):
        """
        Plot train accuracy
        :param skip: int how many steps to skip at the beginning of the plot
        """
        plt.title('Accuracy')
        plt.plot(self.history.history['accuracy'][skip:], label='train')
        plt.plot(self.history.history['val_accuracy'][skip:], label='validation')
        plt.legend()
        plt.show()

    def port(self, arena_size='1024 * 16', model_name='model', classname='NeuralNetwork'):
        """
        Port Tf model to plain C++
        :param arena_size: int|str size of tensor arena (read Tf docs)
        :param model_name: str name of the exported model variable
        :param classname: str name of the exported class
        """
        return jinja('ml/classification/tensorflow/NeuralNetwork.jinja', {
            'classname': classname,
            'model_name': model_name,
            'model_data': port(self.sequential, variable_name=model_name, optimize=False),
            'num_inputs': self.num_inputs,
            'num_outputs': self.num_classes,
            'arena_size': arena_size,
        })

    def to_categorical(self, y):
        """
        One-hot encode y array
        :param y:
        """
        if len(y.shape) == 1 or y.shape[1] == 1:
            return to_categorical(y.astype(np.int).flatten())

        return y