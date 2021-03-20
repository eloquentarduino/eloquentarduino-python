import numpy as np
from eloquentarduino import project as default_project
from eloquentarduino.utils import jinja
#from eloquentarduino.ml.classification.sklearn.SklearnClassifier import SklearnClassifier
#from eloquentarduino.ml.classification.tensorflow.NeuralNetwork import NeuralNetwork


class ClassifierResources:
    """
    Find out how many resources does a classifier requires
    when deployed on device
    """
    def __init__(self, clf, project=None):
        """
        :param clf:
        """
        assert hasattr(clf, 'X') and hasattr(clf, 'y'), 'classifier is unfitted'

        self.clf = clf
        self.project = project or default_project
        self.resources = None

    def get_resources(self):
        """
        Get resources
        """
        if self.resources is None:
            # pick random X elements for benchmark
            num_samples, num_features = self.clf.X.shape[:2]
            pick_samples = 10 if num_features < 10 else 3
            X = self.clf.X[np.random.randint(num_samples, size=pick_samples), :]

            # get resources of a sketch that only prints the dataset
            with self.project.tmp_project() as tmp:
                tmp.files.add('%s.ino' % tmp.name, contents=jinja('on_device/ResourcesBaseline.jinja', {'X': X}), exists_ok=True)
                baseline = tmp.get_resources()

            # get resources for the classifier
            with self.project.tmp_project() as tmp:
                sketch = jinja('on_device/Resources.jinja', {'X': X})
                ported = self.clf.port(classname='Classifier')

                tmp.files.add('%s.ino' % tmp.name, contents=sketch, exists_ok=True)
                tmp.files.add('Classifier.h', contents=ported, exists_ok=True)

                resources = tmp.get_resources()

            self.resources = {
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
