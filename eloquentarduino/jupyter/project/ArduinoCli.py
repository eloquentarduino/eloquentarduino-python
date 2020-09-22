from subprocess import STDOUT, CalledProcessError, check_output


class ArduinoCli:
    """Interact with the Arduino cli"""
    def __init__(self, arguments, autorun=True):
        assert len(arguments) > 0, "ArduinoCli arguments CANNOT be empty"
        self.arguments = arguments
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
        raise RuntimeError(self.error)

    def run(self):
        """Run cli command and save output"""
        try:
            self.output = check_output(['arduino-cli'] + self.arguments, stderr=STDOUT).decode('utf-8')
            self.error = None
        except CalledProcessError as err:
            self.error = err.output.decode('utf-8')
            self.output = None

    def is_successful(self):
        """Test if command was successful"""
        return self.error is None
