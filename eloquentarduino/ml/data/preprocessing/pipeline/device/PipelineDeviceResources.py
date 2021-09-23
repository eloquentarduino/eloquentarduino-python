import numpy as np
from cached_property import cached_property
from eloquentarduino import project as default_project
from eloquentarduino.utils import jinja


class PipelineDeviceResources:
    """
    Benchmark pipeline resources once deployed to a board
    """
    def __init__(self, pipeline, project=None):
        """
        :param pipeline: Pipeline
        """
        self.pipeline = pipeline
        self.project = project or default_project

    @cached_property
    def resources(self):
        """
        Get "static" resource requirements
        :return: dict
        """
        return self._compile()

    @cached_property
    def execution_time(self):
        """
        Get execution time
        :return: float
        """
        # compile and get resources
        resources = self.resources

        return self._execute()

    @property
    def flash_string(self):
        """
        Get readable RAM usage
        """
        return '%d bytes (%d %%)' % (self.resources['flash'], self.resources['flash_percent'] * 100)

    @property
    def ram_string(self):
        """
        Get readable RAM usage
        """
        return '%d bytes (%d %%)' % (self.resources['memory'], self.resources['memory_percent'] * 100)

    def _compile(self):
        """
        Run resource benchmark using arduino-cli
        """
        num_features = self.pipeline.input_dim
        x = np.random.random(num_features)

        pipeline_code = self.pipeline.port(classname='Pipeline', instance_name='pipeline')
        benchmark_code = jinja('on_device/pipeline/Benchmark.ino.jinja', template_data={'x': x})

        with self.project.tmp_project() as project:
            project.files.add('%s.ino' % project.name, contents=benchmark_code, exists_ok=True)
            project.files.add('Pipeline.h', contents=pipeline_code, exists_ok=True)

            return project.get_resources()

    def _execute(self):
        num_features = self.pipeline.input_dim
        x = np.random.random(num_features)

        pipeline_code = self.pipeline.port(classname='Pipeline', instance_name='pipeline')
        benchmark_code = jinja('on_device/pipeline/Benchmark.ino.jinja', template_data={'x': x})

        with self.project.tmp_project() as project:
            project.files.add('%s.ino' % project.name, contents=benchmark_code, exists_ok=True)
            project.files.add('Pipeline.h', contents=pipeline_code, exists_ok=True)
            project.upload()

            return project.serial.read_number('processing time')
