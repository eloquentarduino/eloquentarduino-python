import re
from eloquentarduino.jupyter.project.Errors import BadBoardResponseError
from eloquentarduino.utils import jinja
from eloquentarduino.ml.metrics.device.parsers import CompileLogParser


class Benchmarker:
    """
    Run a single benchmark
    """
    cache = {}

    @classmethod
    def baseline(cls, project, dataset, samples_size=10):
        """
        Benchmark the baseline sketch for the current project and dataset
        """
        cache_key = (project.board.fqbn, dataset.name)

        if cache_key not in cls.cache:
            with project.tmp_project() as tmp:
                X, y = dataset.random(samples_size)
                sketch = jinja('metrics/Baseline.jinja', {'X': X})
                tmp.files.add('%s.ino' % tmp.name, contents=sketch, exists_ok=True)

                cls.cache[cache_key] = CompileLogParser(project=tmp).info

        return cls.cache[cache_key]

    def __init__(self, project, dataset, clf, board=None):
        """
        Constructor
        :param project: Project
        :param dataset: Dataset
        :param clf: Classifier
        :param board: BoardModel
        """
        self.project = project
        self.dataset = dataset
        self.clf = clf
        self.board = board if board is not None else self.project.board

    def get_resources(self, samples_size=10):
        """
        Benchmark the resources for a classifier
        :param samples_size: how many samples to include in the benchmark (should match with the baseline)
        :return: dict resources needed
        """
        with self.project.tmp_project() as tmp:
            template_folder = 'tf' if self.clf.is_tf() else 'sklearn'
            baseline_key = (self.project.board.fqbn, self.dataset.name)

            X, y = self.dataset.random(samples_size)
            sketch = jinja('benchmarks/%s/Resources.jinja' % template_folder, {'X': X})
            ported = self.clf.port(classname='Classifier')

            tmp.files.add('%s.ino' % tmp.name, contents=sketch, exists_ok=True)
            tmp.files.add('Classifier.h', contents=ported, exists_ok=True)

            return CompileLogParser(project=tmp).sub(Benchmarker.cache.get(baseline_key, None)).info

    def get_inference_time(self, samples_size=10, upload_options={}):
        """
        Benchmark onboard inference time for a classifier
        :param samples_size: how many samples to include in the benchmark
        :param upload_options: dict options for upload()
        :return: float inference time in microseconds
        """
        with self.project.tmp_project() as tmp:
            template_folder = 'tf' if self.clf.is_tf() else 'sklearn'

            X, y = self.dataset.random(samples_size)
            sketch = jinja('benchmarks/%s/Runtime.jinja' % template_folder, {'X': X, 'y': y})
            ported = self.clf.port(classname='Classifier')

            tmp.files.add('%s.ino' % tmp.name, contents=sketch, exists_ok=True)
            tmp.files.add('Classifier.h', contents=ported, exists_ok=True)
            tmp.upload(**upload_options)

            # parse serial output
            # since we can miss the first response, try a few times
            for i in range(0, 3):
                response = tmp.serial.read_until('======', timeout=8)
                match = re.search(r'inference time = ([0-9.]+) micros', response)

                if match is not None:
                    return float(match.group(1))

        raise BadBoardResponseError('Unexpected response during runtime inference time benchmark: %s' % response)
