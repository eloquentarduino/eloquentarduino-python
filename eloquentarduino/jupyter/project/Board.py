import re
import os.path
from collections import namedtuple
from eloquentarduino.jupyter.project.ArduinoCli import ArduinoCli


class Board:
    """Interact with the Arduino ecosystem"""
    def __init__(self, project):
        self.project = project
        self.BoardModel = namedtuple('BoardModel', 'name fqbn')
        self.baud_rate = 9600
        self.model = None
        self.port = None

    def assert_model(self):
        """Assert the user set a board model"""
        assert self.model is not None, 'You MUST set a board first'

    def set_model(self, model_pattern):
        """Set board model
            Get the best match from the arduino-cli list of supported boards"""
        # parse known boards from arduino-cli
        known_boards = ArduinoCli(['board', 'listall']).lines
        known_boards = [re.split(r'    +', line) for line in known_boards]
        known_boards = [board for board in known_boards if len(board) == 2]
        known_boards = [self.BoardModel(name=board[0], fqbn=board[1]) for board in known_boards]
        known_boards_names = [board.name for board in known_boards]
        # try exact match on name
        try:
            idx = known_boards_names.index(model_pattern)
            self.model = known_boards[idx]
            self.project.log('Found a match: %s (%s)' % (self.model.name, self.model.fqbn))
            self.project.log('Using it')
        except ValueError:
            # try partial match
            self.project.log('Board [%s] not known, looking for best match...' % model_pattern)
            for i, model in enumerate(self._match_model(known_boards, model_pattern)):
                self.project.log('Found a match: %s (%s)' % (model.name, model.fqbn))
            try:
                # found a single match
                if i == 0:
                    self.model = model
                    self.project.log('Using it')
                else:
                    self.project.log('Please refine your search')
            except UnboundLocalError:
                raise RuntimeError('No match found for board %s' % model_pattern)

    def set_port(self, port):
        """Set port"""
        # if 'auto', search for connected ports
        if port == 'auto':
            available_ports = ArduinoCli(['board', 'list']).lines[1:]
            # if a board has been selected, keep only the lines that match the board
            if self.model is not None:
                available_ports = [line for line in available_ports if self.model.name in line]
            # port is the first column
            available_ports = [line.split(' ')[0] for line in available_ports if ' ' in line]
            assert len(available_ports) > 0, 'No port found'
            # if only one port, use it
            if len(available_ports) == 1:
                port = available_ports[0]
            else:
                # else list them to the user
                for available_port in available_ports:
                    self.project.log('Port found: %s' % available_port)
        self.port = port
        self.project.log('Using port: %s' % self.port)

    def set_baud_rate(self, baud_rate):
        """Set Serial baud rate"""
        assert isinstance(baud_rate, int) and baud_rate > 0, 'Baud rate MUST be a positive integer'
        self.baud_rate = baud_rate
        self.project.log('Set baud rate to', self.baud_rate)

    def compile(self):
        """Compile sketch"""
        self.project.assert_name()
        self.assert_model()
        arguments = ['compile', '--verify', '-b', self.model.fqbn, os.path.abspath(os.path.dirname(self.project.ino_path))]
        self.project.log('arduino-cli', *arguments)
        return ArduinoCli(arguments)

    def upload(self):
        """Upload sketch"""
        self.project.assert_name()
        self.assert_model()
        assert self.port is not None, 'You MUST set a board port'
        arguments = ['upload', '-b', self.model.fqbn, '-p', self.port, os.path.abspath(self.project.ino_path)]
        self.project.log('arduino-cli', *arguments)
        return ArduinoCli(arguments)

    def _match_model(self, known_boards, pattern):
        """Match a model pattern against the known boards"""
        normalizer = re.compile(r'[^a-z0-9 ]')
        pattern = normalizer.sub(' ', pattern.lower())
        pattern_segments = [s for s in pattern.split(' ') if s.strip()]
        for model in known_boards:
            target = normalizer.sub(' ', model.name.lower())
            # it matches if all pattern segments are present in the target
            target_segments = [s for s in target.split(' ') if s.strip()]
            intersection = list(set(pattern_segments) & set(target_segments))
            if len(intersection) == len(pattern_segments):
                yield model