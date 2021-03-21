from copy import copy

from sklearn.model_selection import train_test_split

from eloquentarduino.ml.classification.abstract.GridSearch import GridSearch as GridSearchBase
from eloquentarduino.ml.classification.tensorflow import NeuralNetwork, Layer
from eloquentarduino.ml.classification.tensorflow.gridsearch.GridSearchResult import GridSearchResult


class GridSearch(GridSearchBase):
    """
    Grid search for Tensorflow models
    """
    layers = Layer(None)

    def __init__(self, dataset):
        """
        Constructor
        """
        super().__init__()
        self.dataset = dataset
        self.combinations = [[]]
        self.compile_options = {}
        self.fit_options = {}

    def add_layer(self, layer):
        """
        Add a layer that will always be added to the network
        :param layer:
        """
        assert isinstance(layer, Layer), 'layer MUST be instantiated via GridSearch.layers factory'

        # add layer to all combinations
        new_combinations = []

        for hyper_layer in layer.enumerate():
            new_combinations += [copy(combination) + [copy(hyper_layer)] for combination in self.combinations]

        self.combinations = new_combinations

    def add_optional_layer(self, layer):
        """
        Add a layer that will sometimes be added to the network
        :param layer:
        """
        return self.add_branch([None, layer])

    def add_branch(self, branches):
        """
        Create a branch in the search space for each of the supplied layers
        :param branches: list
        """
        for layer in branches:
            assert layer is None or isinstance(layer, Layer), 'all branches MUST be instantiated via GridSearch.layers factory'

        new_combinations = []

        for layer in branches:
            branch_combinations = []

            if layer is None:
                branch_combinations = [copy(combination) for combination in self.combinations]
            else:
                for hyper_layer in layer.enumerate():
                    branch_combinations += [copy(combination) + [copy(hyper_layer)] for combination in self.combinations]

            new_combinations += branch_combinations

        self.combinations = new_combinations

    def add_softmax(self):
        """
        Add sofmax layer at the end
        """
        self.add_layer(GridSearch.layers.Dense(units=self.dataset.num_classes, activation='softmax'))

    def compile(self, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], **kwargs):
        """
        Set compile options
        """
        self.compile_options = kwargs
        self.compile_options.update(loss=loss, optimizer=optimizer, metrics=metrics)

    def search(self, epochs=30, validation_size=0.2, test_size=0.2, show_progress=True, verbose=0, project=None, **kwargs):
        """
        Perform search
        :param epochs: int
        :param validation_size: float
        :param test_size: float
        :param show_progress: bool
        :param verbose: int
        :param project: Project
        """
        self.results = []

        assert validation_size > 0, 'validation_size MUST be greater than 0'

        self.fit_options = kwargs
        self.fit_options.update(epochs=epochs, verbose=verbose)

        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(self.dataset.X, self.dataset.y_categorical)
        else:
            self.dataset.shuffle()
            X_train, X_test, y_train, y_test = self.dataset.X, None, self.dataset.y, None

        for i, combination in enumerate(self.combinations):
            if show_progress:
                print(i if i % 5 == 0 else '.', end='')

            nn = NeuralNetwork()

            for layer in combination:
                nn.add_layer(copy(layer))

            for key, val in self.compile_options.items():
                nn.set_compile_option(key, val)

            for key, val in self.fit_options.items():
                nn.set_fit_option(key, val)

            nn.fit(X_train, y_train)

            if X_test is None:
                accuracy = max(nn.history.history['val_accuracy'])
            else:
                accuracy = nn.score(X_test, y_test)

            result = GridSearchResult(dataset=self.dataset, clf=nn, accuracy=accuracy)
            self.append_result(result, project=project)

        self.results = sorted(self.results, key=lambda result: result.accuracy, reverse=True)

        return self.results

    def instantiate(self, i=0, fit=True, **kwargs):
        """
        Instantiate result
        :param i: int
        :return: NeuralNetwork
        """
        assert len(self.results) > 0, 'Unfitted'
        assert i < len(self.results), '%d is out of range'

        nn = self.results[i].clf.clone()

        if fit:
            nn.fit(self.dataset.X, self.dataset.y_categorical)

        return nn
