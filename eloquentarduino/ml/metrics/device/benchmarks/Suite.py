import csv
import os.path
from eloquentarduino.ml.data import Dataset, CheckpointFile
from eloquentarduino.ml.classification import Classifier
from eloquentarduino.ml.metrics.device.benchmarks import Benchmarker


class Suite:
    """
    Run a suite of benchmarks
    """
    def __init__(self, project):
        """
        Constructor
        :param project:
        """
        self.project = project
        self.datasets = []
        self.classifiers = []
        self.x_train = None
        self.x_valid = None
        self.x_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
        self.validation_size = 0
        self.shuffle = True
        self.checkpoints = None
        self.existing = None

    def set_datasets(self, datasets, validation_size=0, shuffle=True):
        """
        Set datasets for the suite
        :param datasets: list|Dataset
        :param validation_size: float percent of samples to use as validation
        :param shuffle: bool if dataset should be shuffled before splitting
        """
        self.datasets = datasets if isinstance(datasets, list) else [datasets]
        self.validation_size = validation_size
        self.shuffle = shuffle

        for i, dataset in enumerate(self.datasets):
            assert isinstance(dataset, Dataset), 'dataset[%d] MUST be an instance of Dataset' % i

        assert isinstance(self.validation_size, float), 'validation_size MUST be a float'

    def set_classifiers(self, classifiers):
        """
        Set classifiers for the suite
        :param classifiers: list|Classifier
        """
        self.classifiers = classifiers if isinstance(classifiers, list) else [classifiers]

        for i, classifier in enumerate(self.classifiers):
            assert isinstance(classifier, Classifier), 'classifiers[%d] MUST be an instance of Classifier: instance of %s given' % (i, type(classifier).__name__)

    def save_to(self, filename, existing):
        """
        Save results to a file
        :param filename: string
        :param existing: string what to do if a benchmark already exists. One of {skip, overwrite}
        """
        assert existing in ['skip', 'overwrite'], 'skip MUST be one of {skip, overwrite}'

        self.checkpoints = CheckpointFile(filename, keys=['fqbn', 'dataset', 'clf'])
        self.existing = existing

    def run(self, samples_size=10, cross_validate=3, time=False):
        """
        Run benchmark suite
        :param samples_size: int how many samples to use for resources benchmark. 0 means no resource benchmark
        :param cross_validate: int folds for cross validation accuracy estimate. 0 means no accuracy
        :param time: bool True to benchmark onboard inference time
        """
        num_datasets = len(self.datasets)
        num_classifiers = len(self.classifiers)
        results = []

        # load existing results
        if self.checkpoints is not None and self.existing == 'skip' and os.path.isfile(self.checkpoints.filename):
            with open(self.checkpoints.filename) as file:
                results = list(csv.DictReader(file))

        for i, dataset in enumerate(self.datasets):
            self.project.logger.info('[%d/%d] Benchmarking dataset %s' % (i + 1, num_datasets, dataset.name))

            # benchmark baseline for the classifier
            Benchmarker.baseline(project=self.project, dataset=dataset, samples_size=samples_size)
            self.project.logger.info('Benchmarked baseline')

            for j, clf in enumerate(self.classifiers):
                self.project.logger.info('[%d/%d] Benchmarking classifier %s' % (j + 1, num_classifiers, clf.name))

                key = (self.project.board.fqbn, dataset.name, clf.name)

                if self.checkpoints is not None and self.checkpoints.key_exists(key) and self.existing == 'skip':
                    self.project.logger.info('A checkpoint exists, skipping')
                    continue

                # benchmark accuracy
                accuracy = clf.cross_val_score(dataset, num_folds=cross_validate, validation_size=self.validation_size) if cross_validate > 1 else 0

                # train classifier
                clf.fit(dataset.X, dataset.y, validation_size=self.validation_size)

                result = {
                    'board': self.project.board.name,
                    'fqbn': self.project.board.fqbn,
                    'dataset': dataset.name,
                    'clf': clf.name,
                    'n_features': dataset.num_features,
                    'accuracy': accuracy
                }

                # benchmark resources
                if samples_size > 0:
                    benchmarker = Benchmarker(project=self.project, dataset=dataset, clf=clf)
                    resources = benchmarker.get_resources(samples_size=samples_size)
                    result.update(resources)

                # benchmark onboard inference time
                if time:
                    benchmarker = Benchmarker(project=self.project, dataset=dataset, clf=clf)
                    inference_time = benchmarker.get_inference_time(samples_size=samples_size)
                    result.update(inference_time=inference_time)

                # update results
                results.append(result)

                if self.checkpoints is not None:
                    self.checkpoints.set(key, result)

        return results
