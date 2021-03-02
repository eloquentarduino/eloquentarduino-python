import re
from time import time


class CompileLogParser:
    """
    Extract info from arduino-cli compilation log
    """
    def __init__(self, project=None, compile_log=None):
        assert project is not None or compile_log is not None, 'you MUST set project or compile_log'
        assert compile_log is None or isinstance(compile_log, str), 'compile_log MUST be a string'
        assert project is None or (hasattr(project, 'compile') and callable(project.compile)), 'project MUST implement compile()'

        if compile_log is None:
            start = time()
            compile_log = project.compile()
            compile_time = time() - start
        else:
            compile_time = 0

        flash_pattern = r'Sketch uses (\d+) bytes.+?Maximum is (\d+)'
        memory_pattern = r'Global variables use (\d+).+?Maximum is (\d+)'
        flash_match = re.search(flash_pattern, compile_log.replace("\n", ""))
        memory_match = re.search(memory_pattern, compile_log.replace("\n", ""))

        if flash_match is None and memory_match is None:
            raise RuntimeError('Cannot parse compilation log: %s' % compile_log)

        flash, flash_max = [int(g) for g in flash_match.groups()] if flash_match is not None else [0, 1]
        memory, memory_max = [int(g) for g in memory_match.groups()] if memory_match is not None else [0, 1]

        self.info = {
            'compile_time': compile_time,
            'flash': flash,
            'flash_max': flash_max,
            'flash_percent': float(flash) / flash_max,
            'memory': memory,
            'memory_max': memory_max,
            'memory_percent': float(memory) / memory_max,
        }

    def sub(self, baseline):
        """
        Subtract baseline resources from current ones
        :param baseline: dict of baseline resources
        :return: self
        """
        if not isinstance(baseline, dict):
            return self

        if 'flash' in baseline:
            self.info['flash'] -= baseline['flash']
            self.info['flash_percent'] = float(self.info['flash'] / self.info['flash_max'])

        if 'memory' in baseline:
            self.info['memory'] -= baseline['memory']
            self.info['memory_percent'] = float(self.info['memory'] / self.info['memory_max'])

        return self