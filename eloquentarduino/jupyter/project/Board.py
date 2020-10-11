import re
import os.path
from collections import namedtuple
from shutil import copyfile
from eloquentarduino.jupyter.project.ArduinoCli import ArduinoCli


class Board:
    """Interact with the Arduino ecosystem"""
    def __init__(self, project):
        self.project = project
        self.BoardModel = namedtuple('BoardModel', 'name fqbn')
        self.baud_rate = 9600
        self.cli_path = None
        self.model = None
        self.port = None

    def assert_model(self):
        """Assert the user set a board model"""
        assert self.model is not None, 'You MUST set a board first'

    def set_cli_path(self, folder):
        """Set arduino-cli path"""
        self.cli_path = folder

    def set_model(self, model_pattern):
        """Set board model
            Get the best match from the arduino-cli list of supported boards"""
        # parse known boards from arduino-cli
        lines = self.cli(['board', 'listall']).lines
        board_regex_matches = [re.search(r'^(.+?)\s+([^ :]+?:[^ :]+?:[^ :]+?)$', line) for line in lines]
        known_boards = [self.BoardModel(name=match.group(1), fqbn=match.group(2)) for match in board_regex_matches if match is not None]
        known_boards_names = [board.name for board in known_boards]
        # try exact match on name
        try:
            idx = known_boards_names.index(model_pattern)
            self.model = known_boards[idx]
            self.project.log('Found a match: %s (%s)' % (self.model.name, self.model.fqbn))
            self.project.log('Using it')
        except ValueError:
            # try partial match
            i = -1
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
            available_ports = self.cli(['board', 'list']).lines[1:]
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

    def cli(self, arguments):
        """Execute arduino-cli command"""
        return ArduinoCli(arguments, project=self.project, cli_path=self.cli_path, cwd=self.project.path)

    def self_check(self):
        """Assert that the arduino-cli is working fine"""
        self.cli(['version'])
        return True

    def compile(self):
        """Compile sketch"""
        self.project.assert_name()
        self.assert_model()
        arguments = ['compile', '--verify', '-b', self.model.fqbn]
        ret = self.cli(arguments)
        # hugly hack to make it work with paths containing spaces
        # arduino-cli complains about a "..ino.df" file not found into the build folder
        # so we rename the "{project_name}.dfu" to "..ino.dfu"
        fqbn = self.model.fqbn.replace(':', '.')
        original_file = os.path.abspath(os.path.join(self.project.path, 'build', fqbn, '%s.dfu' % self.project.ino_name))
        hacky_file = os.path.abspath(os.path.join(self.project.path, 'build', fqbn, '..ino.dfu'))
        self.project.log('hacky uploading workaround: renaming %s to %s' % (original_file, hacky_file))
        copyfile(original_file, hacky_file)
        return ret

    def upload(self):
        """Upload sketch"""
        self.project.assert_name()
        self.assert_model()
        assert self.port is not None, 'You MUST set a board port'
        arguments = ['upload', '-b', self.model.fqbn, '-p', self.port, '.']
        return self.cli(arguments)

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