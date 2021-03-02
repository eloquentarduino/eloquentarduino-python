import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tinymlgen import port

from eloquentarduino.ml.data import Dataset
from eloquentarduino.utils import jinja


class TfMicro:
    """
    Eloquent interface to build a NN with Keras and Tf
    """
    def __init__(self, X=None, y=None, dataset=None, fit_config={}):
        """
        :param X:
        :param y:
        :param dataset: Dataset
        :param fit_config: options to apply when you call fit()
        """
        # assert dataset is not None or (X is not None and y is not None), 'you MUST supply a dataset'
        assert dataset is None or isinstance(dataset, Dataset), 'dataset MUST be a eloquent.ml.data.Dataset instance'
        assert isinstance(fit_config, dict), 'fit_config MUST be a dict'

        self.dataset = dataset if dataset is not None else Dataset('dataset', X, y)
        self.fit_config = fit_config
        self.x_train = None
        self.x_test = None
        self.x_validate = None
        self.y_train = None
        self.y_test = None
        self.y_validate = None
        self.sequential = tf.keras.Sequential()
        self.layers = []
        self.history = None

        if len(self.dataset.y.shape) == 1:
            self.dataset.y = to_categorical(self.dataset.y)

    @property
    def num_features(self):
        """
        Get number of features
        :return: int
        """
        return self.dataset.num_features

    @property
    def num_classes(self):
        """
        Get number of classes
        :return: int
        """
        return self.dataset.num_classes

    @property
    def input_shape(self):
        """
        Get input shape
        """
        return self.dataset.X.shape[1:]

    def split(self, train=None, test=None, validation=None, shuffle=True):
        """
        Split dataset into train, test, validation
        :param train: float train size percent
        :param test: float test size percent
        :param validation: float validation size percent
        :param shuffle: bool if X and y should be shuffled before splitting
        :return: TfMicro
        """
        if train is None:
            train = 1 - (test or 0) - (validation or 0)
        if test is None:
            test = 1 - train - (validation or 0)
        if validation is None:
            validation = 1 - train - test

        self.x_train, self.y_train, self.x_validate, self.y_validate, self.x_test, self.y_test = self.dataset.split(test=test, validation=validation, shuffle=shuffle, return_empty=True)

        return self

    def add(self, layer, **kwargs):
        """
        Add layer to network
        :param layer:
        :return: TfMicro
        """
        self.layers.append(layer)

        return self

    def softmax(self, **kwargs):
        """
        Add last dense layer
        :return: TfMicro
        """
        self.add(layers.Dense(self.num_classes, activation='softmax', **kwargs))

        return self.commit()

    def commit(self):
        """
        Add layers to network
        """
        assert len(self.layers) > 0, 'you MUST add at least one layer'

        for layer in self.layers:
            self.sequential.add(layer)

        return self

    def summary(self, *args, **kwargs):
        """
        Return network summary
        """
        return self.sequential.summary(*args, **kwargs)

    def compile(self, optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], **kwargs):
        """
        Compile the network
        :param optimizer:
        :param loss:
        :param metrics:
        """
        return self.sequential.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        """
        Fit network (compatible api with Tf)
        :param X:
        :param y:
        """
        self.sequential.fit(X, y, *args, **kwargs)

    def self_fit(self, **kwargs):
        """
        Fit the network on given dataset
        """
        if self.x_validate is not None and len(self.x_validate) > 0:
            kwargs.update(validation_data=(self.x_validate, self.y_validate))

        if 'validation_data' in kwargs and kwargs['validation_data'][0] is None:
            kwargs.pop('validation_data')

        self.history = self.sequential.fit(self.x_train, self.y_train, **self.fit_config, **kwargs)

        return self.history

    def evaluate(self, x_test=None, y_test=None):
        """
        Evaluate accuracy on given dataset
        :param x_test:
        :param y_test:
        """
        return self.sequential.evaluate(x_test, y_test)

    def self_evaluate(self):
        """
        Evaluate the network on train, test and validation
        :return: (train accuracy, validation accuracy?, test accuracy)
        """
        _, train_acc = self.sequential.evaluate(self.x_train, self.y_train, verbose=0)
        _, test_acc = self.sequential.evaluate(self.x_test, self.y_test, verbose=0)

        if self.x_validate is not None:
            _, validation_acc = self.sequential.evaluate(self.x_validate, self.y_validate, verbose=0)

            return train_acc, validation_acc, test_acc

        return train_acc, test_acc

    def plot(self, loss=True, accuracy=True):
        """
        Plot loss and/or accuracy
        :param loss: bool
        :param accuracy: bool
        """
        plt.figure()

        if loss:
            if accuracy:
                plt.subplot(211)
            plt.title('Loss')
            plt.plot(self.history.history['loss'], label='train')
            plt.plot(self.history.history['val_loss'], label='validation')
            plt.legend()

        if accuracy:
            if loss:
                plt.subplot(212)
            plt.title('Accuracy')
            plt.plot(self.history.history['accuracy'], label='train')
            plt.plot(self.history.history['val_accuracy'], label='validation')
            plt.legend()

        plt.show()

        return self

    def port(self, arena_size='1024 * 16', model_name='model', classname='TfMicro', optimize=False):
        """
        Port Tf model to plain C++
        :param arena_size: size of tensor arena (read Tf docs)
        :param model_name: name of the exported model variable
        :param class_name: name of the exported class
        :param optimize: list of optimizers to apply, or boolean (it is highly recommend to leave this at False)
        """
        return jinja('nn/TfMicro.jinja', {
            'class_name': classname,
            'model_name': model_name,
            'model_data': port(self.sequential, variable_name=model_name, optimize=optimize),
            'num_inputs': self.num_features,
            'num_outputs': self.num_classes,
            'arena_size': arena_size,
            'shape': list(self.dataset.X.shape) + [0]
        })


