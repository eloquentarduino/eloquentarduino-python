import re
import os.path


class SketchFiles:
    """
    Work with project-related files
    """
    def __init__(self, project):
        self.project = project

    def path_to(self, *args):
        """
        Convert relative path to project path
        :param args: path segments
        """
        self.project.assert_name()
        return os.path.abspath(os.path.join('sketches', self.project.name, *args))

    def mkdir(self, dirname):
        """
        Create folder inside project
        :param dirname: folder path
        """
        self.project.logger.debug('Adding folder %s to project', self.path_to(dirname))
        os.makedirs(self.path_to(dirname), exist_ok=True)

    def open(self, *args, mode='r'):
        """
        Open file inside project folder
        :param args: path segments
        :param mode: file mode
        """
        self.project.logger.debug('Opening file %s in %s mode', self.path_to(*args), mode)
        return open(self.path_to(*args), mode=mode)

    def add(self, *args, contents, exists_ok=False):
        """
        Write contents to a project file
        :param args: path segments
        :param contents: contents to write
        :param exists_ok: wether to overwrite existing contents
        :return:
        """
        filename = self.path_to(*args)
        self.project.logger.debug('Writing to file %s', filename)
        self.mkdir(os.path.dirname(os.path.join(*args)))
        # prevent overwriting existing file
        if os.path.exists(filename) and not exists_ok:
            self.project.logger.warning('File already exists... skipping')
            return False
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(contents)

    def cat(self, *args):
        """
        Return the contents of a project file.
        Defaults to .ino file
        """
        if len(args) == 0:
            args = ['%s.ino' % self.project.name]

        self.project.logger.debug('Reading contents of file %s', self.path_to(*args))

        with open(self.path_to(*args), encoding="utf-8") as file:
            return file.read()

    def find_replace(self, *args, find=None, replace=None, count=9999):
        """
        Find and replace contents in file
        :param find: str
        :param replace: str
        :param count: int
        """
        assert find is not None, 'find MUST be set'
        assert replace is not None, 'replace MUST be set'

        contents = self.cat(*args).replace(find, replace, count)
        self.add(*args, contents=contents, exists_ok=True)

    def find_replace_regex(self, *args, find=None, replace=None, count=9999):
        """
        Find and replace contents in file
        :param find: str
        :param replace: str
        :param count: int
        """
        assert find is not None, 'find MUST be set'
        assert replace is not None, 'replace MUST be set'

        contents = re.sub(find, replace, self.cat(*args), count)
        self.add(*args, contents=contents, exists_ok=True)
