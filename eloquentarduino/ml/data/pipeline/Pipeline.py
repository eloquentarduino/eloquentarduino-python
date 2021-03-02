from collections import namedtuple


PipelineStep = namedtuple('PipelineStep', 'name step')


class Pipeline:
    """
    Define a pre-processing pipeline that can be ported to plain C
    """
    def __init__(self, name, X, y):
        self.name = name
        self.X = X
        self.y = y
        self.steps = []

    def add(self, step_name, step):
        self.X, self.y = step.apply(self.X, self.y)
        self.steps.append(PipelineStep(name=step_name, step=step))
