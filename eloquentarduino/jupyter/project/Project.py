import os
import os.path
from datetime import datetime
from time import sleep

from eloquentarduino.jupyter.project.Board import Board
from eloquentarduino.jupyter.project.CompileStatistics import CompileStatistics
from eloquentarduino.jupyter.project.Serial import SerialMonitor
from eloquentarduino.jupyter.project.SketchFiles import SketchFiles


class Project:
    """Interact programmatically with an Arduino project"""
    def __init__(self):
        self._name = ''
        self.board = Board(self)
        self.serial = SerialMonitor(self)
        self.files = SketchFiles(self)
        self.compile_statistics = None
        self.ml_classifiers = []

    @property
    def name(self):
        """Get name"""
        return self._name

    @property
    def path(self):
        """Get path to sketch directory"""
        return os.path.join('sketches', self.name)

    @property
    def ino_name(self):
        """Get name of .ino file"""
        return '%s.ino' % self.name

    @property
    def ino_path(self):
        """Get path to .ino file"""
        return os.path.join(self.path, self.ino_name)

    def assert_name(self):
        """Assert the user set a project name"""
        assert self.name, 'You MUST set a project name'

    def log(self, *args, **kwargs):
        """Log info to console"""
        print(*args, **kwargs)

    def set_default_name(self, suffix):
        """Set name according to the Arduino default policy"""
        now = datetime.now()
        sketch_name = now.strftime('sketch_%a%d').lower() + suffix
        self.set_name(sketch_name)

    def set_name(self, name):
        """Set project name. Create a folder if it does not exist"""
        assert isinstance(name, str) and len(name) > 0, 'Sketch name CANNOT be empty'
        self._name = name
        self.log('Set project name', self._name)
        # make project folders (sketch, data)
        self.files.mkdir('')
        self.files.mkdir('data')

    def set_arduino_cli_path(self, folder):
        """Set arduino-cli path"""
        self.log('set arduino-cli path to', folder)
        self.board.set_cli_path(folder)

    def compile(self):
        """Compile sketch using arduino-cli"""
        command = self.board.compile()
        self.log(command.safe_output)
        # hack to allow path with spaces
        if command.is_successful():
            self.compile_statistics = CompileStatistics(command.output)

    def upload(self):
        """Upload sketch using arduino-cli"""
        self.compile()
        command = self.board.upload()
        self.log(command.safe_output)
        sleep(1)

    # def port(self, clf):
    #     """Add Python ML classifier to current project"""
    #     ported = port(clf)
    #     classifier_name = ported.split('class ')[1].split(' ')[0]
    #     filename = '%s.h' % classifier_name
    #     self.log('Saving ported classifier to %s' % filename)
    #     with self.open(filename, 'w') as file:
    #         file.write(ported)


# singleton instance
project = Project()