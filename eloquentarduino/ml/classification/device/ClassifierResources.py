import re
import numpy as np
from math import ceil
from eloquentarduino import project as default_project
from eloquentarduino.utils import jinja


class ClassifierResources:
    """
    Find out how many resources does a classifier requires
    when deployed on device
    """
    # keep a record of resources for baseline sketches
    # which only depends on the number of samples * number of features
    _baseline_cache = {}

    def __init__(self, clf, project=None):
        """
        :param clf:
        """
        assert hasattr(clf, 'X') and hasattr(clf, 'y'), 'classifier is unfitted'

        self.clf = clf
        self.project = project or default_project
        self.resources = None
        self.inference_time = None

    @property
    def is_sklearn(self):
        """
        Test if classifier is sklearn
        """
        return 'sklearn' in self.clf.__module__

    @property
    def is_tensorflow(self):
        """
        Test if classifier is tensorflow
        """
        return 'tensorflow' in self.clf.__module__

    @property
    def package(self):
        """
        Get package for templates
        """
        if self.is_sklearn:
            return 'sklearn'
        if self.is_tensorflow:
            return 'tensorflow'
        raise AssertionError('classifier is neither sklearn nor tensorflow')

    def get_resources(self, force_update=False):
        """
        Get resources
        :param force_update: bool if to force a new computation
        """
        if self.resources is None or force_update:
            X = self.pick_random()
            pick_samples, num_features = X.shape[:2]

            # get resources of a sketch that only prints the dataset
            cache_key = pick_samples * num_features

            if cache_key not in ClassifierResources._baseline_cache:
                with self.project.tmp_project() as tmp:
                    tmp.files.add('%s.ino' % tmp.name, contents=self.jinja('ResourcesBaseline.jinja', X=X), exists_ok=True)
                    baseline = tmp.get_resources()
                    ClassifierResources._baseline_cache[cache_key] = baseline

            baseline = ClassifierResources._baseline_cache[cache_key]

            # get resources for the classifier
            with self.project.tmp_project() as tmp:
                sketch = self.jinja('Resources.jinja', X=X)
                ported = self.clf.port(classname='Classifier')

                tmp.files.add('%s.ino' % tmp.name, contents=sketch, exists_ok=True)
                tmp.files.add('Classifier.h', contents=ported, exists_ok=True)

                resources = tmp.get_resources()

            self.resources = {
                'compile_time': resources['time'],
                'flash': resources['flash'],
                'rel_flash': resources['flash'] - baseline['flash'],
                'flash_max': resources['flash_max'],
                'flash_percent': resources['flash'] / resources['flash_max'],
                'rel_flash_percent': (resources['flash'] - baseline['flash']) / resources['flash_max'],
                'memory': resources['memory'],
                'rel_memory': resources['memory'] - baseline['memory'],
                'memory_max': resources['memory_max'],
                'memory_percent': resources['memory'] / resources['memory_max'],
                'rel_memory_percent': (resources['memory'] - baseline['memory']) / resources['memory_max']
            }

        return self.resources

    def get_inference_time(self, force_update=False, upload_options={}):
        """
        Get inference time
        :param force_update: bool if to force a new computation
        :param upload_options: dict options for the upload() method
        """
        if self.inference_time is None or force_update:
            X = self.pick_random()

            with self.project.tmp_project() as tmp:
                sketch = self.jinja('InferenceTime.jinja', X=X)
                ported = self.clf.port(classname='Classifier')

                tmp.files.add(tmp.ino_name, contents=sketch, exists_ok=True)
                tmp.files.add('Classifier.h', contents=ported, exists_ok=True)
                tmp.upload(**upload_options)

                # parse serial output
                # since we can miss the first response, try a few times
                for i in range(0, 5):
                    response = tmp.serial.read_until('======', timeout=8)
                    match = re.search(r'inference time = ([0-9.]+) micros', response)

                    if match is not None:
                        self.inference_time = float(match.group(1))
                        break

        return self.inference_time

    def pick_random(self, max_elements=300):
        """
        Pick random X elements for benchmark
        :param max_elements: int max size of num_samples * num_features
        """
        num_samples, num_features = self.clf.X.shape[:2]
        pick_samples = ceil(max_elements // num_features)

        return self.clf.X[np.random.randint(num_samples, size=pick_samples)]

    def jinja(self, template_name, **kwargs):
        """
        Get jinja template from current package
        """
        return jinja('on_device/%s/%s' % (self.package, template_name), kwargs)
