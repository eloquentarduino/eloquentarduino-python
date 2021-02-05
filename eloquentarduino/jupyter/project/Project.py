import os
import re
import os.path
from copy import copy
from datetime import datetime
from time import sleep

from eloquentarduino.jupyter.project.Board import Board
from eloquentarduino.jupyter.project.Errors import UploadNotVerifiedError, ArduinoCliCommandError
from eloquentarduino.jupyter.project.Logger import ProjectLogger
from eloquentarduino.jupyter.project.SerialMonitor import SerialMonitor
from eloquentarduino.jupyter.project.SketchFiles import SketchFiles


class Project:
    """
    Interact programmatically with an Arduino project
    """

    def __init__(self):
        self._name = ''
        self.board = Board(self)
        self.serial = SerialMonitor(self)
        self.files = SketchFiles(self)
        self.logger = ProjectLogger('eloquentarduino.jupyter.Project')

    def __enter__(self):
        """
        Synctactic sugar
        :return:
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Synctactic sugar
        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        pass

    @property
    def name(self):
        """
        Get project name
        """
        return self._name

    @property
    def path(self):
        """
        Get path to sketch directory
        """
        return os.path.join('sketches', self.name)

    @property
    def ino_name(self):
        """
        Get name of .ino file
        """
        return '%s.ino' % self.name

    @property
    def ino_path(self):
        """
        Get path to .ino file
        """
        return os.path.join(self.path, self.ino_name)

    def assert_name(self):
        """
        Assert the user set a project name
        """
        assert self.name, 'You MUST set a project name'

    def log(self, *args, **kwargs):
        """
        Log info to console
        @deprecated
        """
        print(*args, **kwargs)

    def set_default_name(self, suffix):
        """
        Set name according to the Arduino default policy
        """
        now = datetime.now()
        sketch_name = now.strftime('sketch_%a%d').lower() + suffix
        self.set_name(sketch_name)

    def set_name(self, name):
        """
        Set project name
        Create a folder if it does not exist
        """
        assert isinstance(name, str) and len(name) > 0, 'Sketch name CANNOT be empty'
        self._name = name
        self.logger.info('Set project name to %s', self._name)
        # make project folders (sketch, data)
        self.files.mkdir('')
        self.files.mkdir('data')

    def set_arduino_cli_path(self, folder):
        """Set arduino-cli path"""
        self.logger.info('set arduino-cli path to %s', folder)
        self.board.set_cli_path(folder)

    def tmp_project(self):
        """
        Clone project with a temporary directory
        :return: Project
        """
        tmp = Project()
        tmp.set_name('tmp')
        tmp.board = copy(self.board)
        tmp.board.project = tmp
        tmp.serial = copy(self.serial)
        tmp.serial.project = tmp
        tmp.files = copy(self.files)
        tmp.files.project = tmp

        return tmp

    def compile(self):
        """
        Compile sketch using arduino-cli
        :return:
        """
        output = self.board.compile().safe_output
        self.logger.debug('Compile log\n%s', output)
        self.logger.info('Compile OK')
        return output

    def upload(self, compile=True, retry=True, success_message=r'ok|verified|done|found|success', wait_for=2):
        """
        Upload sketch using arduino-cli
        :param compile: wether to compile the sketch before uploading
        :param retry: wether to retry the upload on failure
        :param success_message: string to look for to assert the upload was successful
        :param wait_for:
        :return:
        """
        if compile:
            self.compile()

        try:
            # run upload
            command = self.board.upload()
            output = command.safe_output
            self.logger.debug('Upload output\n%s', output)
        except ArduinoCliCommandError as err:
            # if error, ask the user to reset the board
            self.logger.error(err)
            if retry:
                input('arduino-cli returned an error: try to un-plug and re-plug the board, then press Enter...')

                return self.upload(compile=False, retry=False)
            else:
                # if it errored even after resetting, abort
                raise err

        # assert upload is ok
        if re.search(success_message, output.lower()) is None:
            self.logger.warning('Cannot find success message in log')
            self.logger.debug(output)
            if retry:
                input('Verification failed: try to un-plug and re-plug the board, then press Enter...')

                return self.upload(compile=False, retry=False)
            else:
                raise UploadNotVerifiedError()

        self.logger.info('Upload OK')
        sleep(wait_for)
        return output


# singleton instance
project = Project()