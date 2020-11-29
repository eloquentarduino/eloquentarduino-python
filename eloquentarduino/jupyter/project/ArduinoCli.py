import os
import os.path
from platform import system
from subprocess import STDOUT, CalledProcessError, check_output

from eloquentarduino.jupyter.project.Errors import ArduinoCliCommandError


class ArduinoCli:
    """Interact with the Arduino cli"""
    def __init__(self, arguments, project, autorun=True, cli_path=None, cwd=None):
        assert len(arguments) > 0, "ArduinoCli arguments CANNOT be empty"
        self.arguments = arguments
        self.project = project
        self.cli_path = cli_path
        self.cwd = os.path.abspath(cwd) if cwd is not None else None
        self.output = None
        self.error = None
        if autorun:
            self.run()

    @property
    def lines(self):
        """Get command output as lines"""
        assert self.output is not None, "cannot get lines of errored command"
        return [line.strip() for line in self.output.split("\n")]

    @property
    def safe_output(self):
        """Get output if ok, else raise error"""
        if self.is_successful():
            return self.output
        raise ArduinoCliCommandError(self.error)

    @property
    def executable(self):
        """Return command line executable"""
        executable = 'arduino-cli.exe' if 'window' in system().lower() else 'arduino-cli'
        if self.cli_path is None:
            return executable
        return os.path.join(self.cli_path, executable)

    def run(self):
        """Run cli command and save output"""
        try:
            self.project.logger.debug('(cwd %s) %s %s', self.cwd, self.executable, ' '.join([str(arg) for arg in self.arguments]))
            self.output = check_output([self.executable] + self.arguments, stderr=STDOUT, cwd=self.cwd).decode('utf-8')
            self.error = None
        except CalledProcessError as err:
            self.error = err.output.decode('utf-8')
            self.output = None

    def is_successful(self):
        """Test if command was successful"""
        return self.error is None
