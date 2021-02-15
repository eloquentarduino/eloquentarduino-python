import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from eloquentarduino.ml.data import Dataset


class TfMicro:
    """
    Eloquent interface to build a NN with Keras and Tf
    """
    def __init__(self, X=None, y=None, dataset=None):
        """
        :param X:
        :param y:
        :param dataset: a eloquentarduino.ml.data.Dataset instance
        """
        assert dataset is not None or (X is not None and y is not None), 'you MUST supply a dataset'
        assert dataset is None or isinstance(dataset, Dataset), 'dataset MUST be a eloquent.ml.data.Dataset instance'

        self.dataset = dataset if dataset is not None else Dataset('dataset', X, y)
        self.x_train = None
        self.x_test = None
        self.x_validate = None
        self.y_train = None
        self.y_test = None
        self.y_validate = None
        self.sequential = tf.keras.Sequential()
        self.layers = []
        self.history = None

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

        assert 0 < train < 1, 'train size MUST be in the range (0, 1)'
        assert 0 < test < 1, 'test size MUST be in the range (0, 1)'
        assert 0 <= validation < 1, 'validation size MUST be in the range [0, 1)'

        train_split = int(train * self.dataset.length)
        test_split = int(test * self.dataset.length) + train_split

        if shuffle:
            self.dataset.shuffle()

        if validation > 0:
            self.x_train, self.x_test, self.x_validate = np.split(self.dataset.X, [train_split, test_split])
            self.y_train, self.y_test, self.y_validate = np.split(self.dataset.y, [train_split, test_split])
        else:
            self.x_train, self.x_test = np.split(self.dataset.X, [train_split])
            self.y_train, self.y_test = np.split(self.dataset.y, [train_split])

        return self

    def add(self, layer, **kwargs):
        """
        Add layer to network
        :param layer:
        :return: TfMicro
        """
        self.layers.append(layer)

        return self

    def last_dense(self, **kwargs):
        """
        Add last dense layer
        :return: TfMicro
        """
        self.add(layers.Dense(self.num_classes, **kwargs))

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

    def compile(self, **kwargs):
        """
        Compile the network
        """
        return self.sequential.compile(**kwargs)

    def fit(self, *args, **kwargs):
        """
        Fit the network
        """
        if self.x_validate is not None:
            kwargs.update(validation_data=(self.x_validate, self.y_validate))

        self.history = self.sequential.fit(self.x_train, self.y_train, *args, **kwargs)

        return self.history

    def evaluate(self):
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

