import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from copy import copy
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tinymlgen import port
from functools import reduce
from eloquentarduino.utils import jinja
from eloquentarduino.ml.classification.abstract.Classifier import Classifier
from eloquentarduino.ml.classification.tensorflow.Layer import Layer, layers
from eloquentarduino.ml.classification.device import ClassifierResources


class NeuralNetwork(Classifier):
    """
    Tensorflow neural network abstraction
    """
    def __init__(self, compile_options={}, fit_options={}, name=''):
        """
        :param compile_options: dict
        :param fit_options: dict
        :param name: str
        """
        self.name = name
        self.layer_definitions = []
        self.X = None
        self.y = None
        self.compile_options = {
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy'],
            'optimizer': 'adam',
        }
        self.fit_options = {
            'epochs': 20,
            'valid_size': 0.2,
            'batch_size': 32
        }
        self.compile_options.update(compile_options)
        self.fit_options.update(fit_options)
        self.sequential = None
        self.history = None
        self.reset()

    def __str__(self):
        """
        Get string representation
        """
        return '%s (%s)' % (self.name or 'NeuralNetwork', str(self.describe()))

    def __repr__(self):
        """
        Get string representation
        """
        return str(self)

    @property
    def num_inputs(self):
        return reduce(lambda x, prod: x * prod, self.X.shape[1:], 1)

    @property
    def num_classes(self):
        return self.y.shape[1]

    @property
    def train_accuracy(self):
        """
        Get train accuracy
        """
        accuracy = self.history.history.get('val_accuracy', self.history.history['accuracy'])

        return accuracy[-1]

    def clone(self):
        """
        Clone current network
        """
        nn = NeuralNetwork(compile_options=self.compile_options, fit_options=self.fit_options)
        nn.layer_definitions = [copy(layer) for layer in self.layer_definitions]
        nn.X = self.X
        nn.y = self.y

        return nn

    def reset(self):
        """
        Reset the network
        """
        self.sequential = None
        self.history = None

    def add_dense(self, *args, **kwargs):
        """
        Add dense layer
        """
        return self.add_layer(layers.Dense(*args, **kwargs))

    def add_softmax(self):
        """
        Add softmax layer
        """
        return self.add_dense(units='num_classes', activation='softmax')

    def add_conv2d(self, *args, **kwargs):
        """
        Add Conv2D layer
        """
        return self.add_layer(layers.Conv2D(*args, **kwargs))

    def add_flatten(self):
        """
        Add flatten layer
        """
        return self.add_layer(layers.Flatten())

    def add_layer(self, layer, *args, **kwargs):
        """
        Add generic layer
        :param layer: layers.Layer
        """
        assert isinstance(layer, Layer), 'layer MUST be instantiated via the eloquentarduino.ml.classification.tensorflow.layers factory'

        self.layer_definitions.append(layer)

        return self

    def validate_on(self, percent):
        """
        Set validation percent size
        :param percent: float
        """
        self.fit_options['valid_size'] = percent

    def set_epochs(self, epochs):
        """
        Set number of epochs for training
        :param epochs: int
        """
        self.fit_options['epochs'] = epochs

    def set_optimizer(self, optimizer):
        """
        Set optimizer
        :param optimizer: str
        """
        if optimizer is None:
            return

        self.compile_options['optimizer'] = optimizer

    def set_loss(self, loss):
        """
        Set loss function
        :param loss: str
        """
        if loss is None:
            return
        self.compile_options['loss'] = loss

    def set_metrics(self, metrics):
        """
        Set metrics
        :param metrics: list
        """
        if metrics is None:
            return
        self.compile_options['metrics'] = metrics

    def set_batch_size(self, batch_size):
        """
        Set batch_size
        :param batch_size: int
        """
        self.fit_options['batch_size'] = batch_size

    def set_compile_option(self, key=None, value=None, **kwargs):
        """
        Set compile option
        """
        if key is not None:
            self.compile_options[key] = value

        self.compile_options.update(kwargs)

    def set_fit_option(self, **kwargs):
        """
        Set fit options
        """
        self.fit_options.update(**kwargs)

    def fit(self, X, y, **kwargs):
        """
        Fit the network
        :param X:
        :param y:
        """
        self.sequential = tf.keras.Sequential()
        y = self.to_categorical(y)

        self.fit_options.update(kwargs)

        # build the network
        for i, layer_definition in enumerate(self.layer_definitions):
            kwargs = layer_definition.kwargs

            if kwargs.get('units', 0) == 'num_classes':
                kwargs['units'] = y.shape[1]

            if i == 0 and 'input_shape' not in kwargs:
                kwargs['input_shape'] = X.shape[1:]

            self.sequential.add(layer_definition.instantiate())

        validation_data = None

        if self.fit_options.get('valid_size', 0) > 0:
            X, X_valid, y, y_valid = train_test_split(X, y, test_size=self.fit_options['valid_size'])
            validation_data = (X_valid, y_valid)
            del self.fit_options['valid_size']

        self.sequential.compile(**self.compile_options)
        self.history = self.sequential.fit(X, y, validation_data=validation_data, **self.fit_options)
        self.X = X
        self.y = y

        return self

    def predict(self, X):
        """
        Predict
        :param X:
        """
        return self.sequential.predict(X)

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

    def describe(self):
        """
        Get layers description
        """
        return {
            'layers': self.layer_definitions,
            'compile_options': self.compile_options,
            'fit_options': self.fit_options
        }

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

        if 'val_accuracy' in self.history.history:
            plt.plot(self.history.history['val_accuracy'][skip:], label='validation')

        plt.legend()
        plt.show()

    def port(self, arena_size='1024 * 16', model_name='model', classname='NeuralNetwork', classmap=None):
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
            'classmap': classmap
        })

    def on_device(self, project=None):
        """
        Get device benchmarker
        :param project: Project
        """
        return ClassifierResources(self, project=project)

    def to_categorical(self, y):
        """
        One-hot encode y array
        :param y:
        """
        if len(y.shape) == 1 or y.shape[1] == 1:
            return to_categorical(y.astype(np.int).flatten())

        return y
