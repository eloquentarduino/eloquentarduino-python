import re
from collections import namedtuple

from eloquentarduino.jupyter.project.ArduinoCli import ArduinoCli
from eloquentarduino.jupyter.project.Errors import BoardNotFoundError, MultipleBoardsFoundError, NoSerialPortFoundError, MultipleSerialPortsFoundError


class Board:
    """
    Interact with the Arduino ecosystem via arduino-cli
    """
    def __init__(self, project):
        self.project = project
        self.BoardModel = namedtuple('BoardModel', 'name fqbn')
        self.baud_rate = 9600
        self.cli_path = None
        self.model = None
        self.port = None
        self.programmer = None

    def set_cli_path(self, folder):
        """
        Set arduino-cli path
        """
        self.cli_path = folder

    def self_check(self):
        """
        Assert that the arduino-cli is working fine
        :return: the version of arduino-cli
        """
        return self.cli(['version']).safe_output

    def list_all(self):
        """
        Get list of installed boards from arduino-cli
        :return:
        """
        lines = self.cli(['board', 'listall']).lines
        return list(filter(lambda x: x is not None, [self._parse(line) for line in lines]))

    def set_model(self, model_pattern):
        """
        Set board model
        :param model_pattern: board name or FQBN, either exact or partial
        """
        known_boards = self.list_all()
        # look for exact match on name or fqbn
        matches = [board for board in known_boards if board.name == model_pattern or board.fqbn == model_pattern]

        if len(matches) == 1:
            self.model = matches[0]
            self.project.logger.info('Found an exact match: %s (%s). Using it', self.model.name, self.model.fqbn)

        # look for partial match
        matches = [board for board in known_boards if self._matches(board, model_pattern)]

        if len(matches) == 0:
            raise BoardNotFoundError('Board %s not found in the list of known boards', model_pattern)
        elif len(matches) == 1:
            self.model = matches[0]
            self.project.logger.info('Found a single partial match: %s (%s). Using it', self.model.name, self.model.fqbn)
        else:
            for match in matches:
                self.project.logger.debug('Found a match: %s', match)
            raise MultipleBoardsFoundError('Multiple boards match the given pattern, please refine the search')

    def set_port(self, port):
        """
        Set board serial port
        :param port:
        :return:
        """
        # if port==auto, auto-detect port
        if port == 'auto':
            available_ports = [line for line in self.cli(['board', 'list']).lines[1:] if line.strip()]

            if len(available_ports) == 0:
                raise NoSerialPortFoundError('Cannot find any serial port connected')

            for available_port in available_ports:
                self.project.logger.debug('Serial port found "%s"', available_port)

            # if a board has been selected, keep only the lines that match the board
            if self.model is not None:
                available_ports = [line for line in available_ports if self.model.name in line]

                if len(available_ports) == 0:
                    raise NoSerialPortFoundError('Cannot find any serial port connected for the board %s' % self.model.name)

            # port name is the first column
            available_ports = [line.split(' ')[0] for line in available_ports if ' ' in line]

            if len(available_ports) == 0:
                raise NoSerialPortFoundError('Cannot find any candidate serial port connected')

            # if only one port has been found, use it
            if len(available_ports) == 1:
                port = available_ports[0]
                self.project.logger.debug('Single matching port found, using it')
            else:
                raise MultipleSerialPortsFoundError('Found multiple serial ports, please don\'t use "auto"')

        self.port = port
        self.project.logger.info('Using port %s', self.port)

    def set_baud_rate(self, baud_rate):
        """
        Set serial baud rate
        :param baud_rate:
        :return:
        """
        assert isinstance(baud_rate, int) and baud_rate > 0, 'Baud rate MUST be a positive integer'
        self.baud_rate = baud_rate
        self.project.logger.info('Set baud rate to %d', self.baud_rate)

    def set_programmer(self, programmer):
        """
        Set board programmer
        :param programmer:
        :return:
        """
        self.programmer = programmer
        return self

    def cli(self, arguments):
        """Execute arduino-cli command"""
        return ArduinoCli(arguments, project=self.project, cli_path=self.cli_path, cwd=self.project.path)

    def compile(self):
        """
        Compile sketch
        """
        self._assert(port=False)
        return self.cli(['compile', '--verify', '--fqbn', self.model.fqbn])

        # @todo still not working in some cases
        # hugly hack to make it work with paths containing spaces
        # arduino-cli complains about a "..ino.df" file not found into the build folder
        # so we rename the "{project_name}.dfu" to "..ino.dfu"
        # fqbn = self.model.fqbn.replace(':', '.')
        # original_file = os.path.abspath(os.path.join(self.project.path, 'build', fqbn, '%s.dfu' % self.project.ino_name))
        # if os.path.isfile(original_file):
        #     hacky_file = os.path.abspath(os.path.join(self.project.path, 'build', fqbn, '..ino.dfu'))
        #     self.project.logger.debug('hacky uploading workaround: renaming %s to %s' % (original_file, hacky_file))
        #     copyfile(original_file, hacky_file)
        # return ret

    def upload(self):
        """Upload sketch"""
        self._assert(port=True)
        arguments = ['upload', '--verify', '--fqbn', self.model.fqbn, '--port', self.port]

        if self.programmer:
            arguments += ['--programmer', self.programmer]

        return self.cli(arguments)

    def _assert(self, port=False):
        """
        Assert that everything is configured properly
        :param port: assert that port is configured
        :return:
        """
        self.project.assert_name()
        assert self.model is not None, 'You MUST set a board'
        assert port is False or self.port is not None, 'You MUST set a board port'

    def _parse(self, line):
        """
        Parse arduino-cli board definition line
        :param line: an arduino-cli board definition line
        :return: the parsed BoardModel, or None on error
        """
        match = re.search(r'^(.+?)\s+([^ :]+?:[^ :]+?:[^ :]+?)$', line)

        if match is not None:
            return self.BoardModel(name=match.group(1), fqbn=match.group(2))

        return None

    def _matches(self, model, pattern):
        """
        Test if a model pattern matches against a board
        :param model: a BoardModel instance
        :param pattern: a string pattern
        :return:
        """
        normalizer = re.compile(r'[^a-z0-9 ]')
        pattern = normalizer.sub(' ', pattern.lower())
        pattern_segments = [s for s in pattern.split(' ') if s.strip()]
        target = normalizer.sub(' ', '%s %s' % (model.name.lower(), model.fqbn.lower()))
        # it matches if all pattern segments are present in the target
        target_segments = [s for s in target.split(' ') if s.strip()]
        intersection = list(set(pattern_segments) & set(target_segments))

        return len(intersection) == len(pattern_segments)