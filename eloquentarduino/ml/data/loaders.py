import numpy as np
from os.path import basename, splitext
from glob import glob


def load_from_directory(directory, ext="csv", delimiter=","):
    """Load data from a directory.
        Each file is a class, each line is a sample.
        :returns
            - X: matrix of samples
            - y: array of labels
            - classmap: dict {class_idx: class_name}
    """
    dataset = None
    classmap = {}
    for class_idx, filename in enumerate(glob("%s/*.%s" % (directory, ext))):
        label = splitext(basename(filename))[0]
        classmap[class_idx] = label
        X = np.loadtxt(filename, dtype=np.float, delimiter=delimiter)
        y = np.ones((len(X), 1)) * class_idx
        samples = np.hstack((X, y))
        dataset = samples if dataset is None else np.vstack((dataset, samples))
    return dataset[:, :-1], dataset[:, -1], classmap
