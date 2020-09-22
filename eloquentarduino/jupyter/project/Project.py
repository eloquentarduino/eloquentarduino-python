import os
import os.path
from datetime import datetime
from time import sleep
from micromlgen import port
from eloquentarduino.jupyter.project.Board import Board
from eloquentarduino.jupyter.project.Serial import SerialMonitor
from eloquentarduino.jupyter.project.CompileStatistics import CompileStatistics


class Project:
    """Interact programmatically with an Arduino project"""
    def __init__(self):
        self._name = ''
        self.board = Board(self)
        self.serial = SerialMonitor(self)
        self.compile_statistics = None
        self.ml_classifiers = []

    @property
    def name(self):
        """Get name"""
        return self._name

    @property
    def path(self):
        """Get path to sketch directory"""
        return os.path.join("sketches", self.name)

    @property
    def ino_path(self):
        """Get path to .ino file"""
        return os.path.join(self.path, "%s.ino" % self.name)

    def assert_name(self):
        """Assert the user set a project name"""
        assert self.name, "You MUST set a project name"

    def log(self, *args, **kwargs):
        """Log info"""
        print(*args, **kwargs)

    def sketches_path(self, *path):
        """Return path to sketches folder"""
        return os.path.join("sketches", *path)

    def open(self, filename, mode="r"):
        """Open file in current project"""
        self.log("Opening file %s in `%s` mode" % (filename, mode))
        return open(self.sketches_path(self.name, filename), mode, encoding="utf-8")

    def set_default_name(self, suffix):
        """Set name according to the Arduino default policy"""
        now = datetime.now()
        sketch_name = now.strftime("sketch_%a%d").lower() + suffix
        self.set_name(sketch_name)

    def set_name(self, name):
        """Set project name. Create a folder if it does not exist"""
        assert isinstance(name, str) and len(name) > 0, "Sketch name CANNOT be empty"
        self._name = name
        self.log("Set project name", self._name)
        # make project folders (sketch, data)
        os.makedirs(self.sketches_path(self.name), exist_ok=True)
        os.makedirs(self.sketches_path(self.name, "data"), exist_ok=True)


    def cat(self):
        """Print sketch contents"""
        assert self.name is not None, "You MUST set a project name"
        with open(self.ino_path, encoding="utf-8") as file:
            self.log(file.read())

    def compile(self):
        """Compile sketch using arduino-cli"""
        command = self.board.compile()
        self.log(command.safe_output)
        if command.is_successful():
            self.compile_statistics = CompileStatistics(command.output)

    def upload(self):
        """Upload sketch using arduino-cli"""
        command = self.board.upload()
        self.log(command.safe_output)
        sleep(1)

    def port(self, clf):
        """Add Python ML classifier to current project"""
        ported = port(clf)
        classifier_name = ported.split('class ')[1].split(' ')[0]
        filename = "%s.h" % classifier_name
        self.log("Saving ported classifier to %s" % filename)
        with self.open(filename, "w") as file:
            file.write(ported)


# singleton instance
project = Project()