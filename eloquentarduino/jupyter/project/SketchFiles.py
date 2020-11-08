import os.path


class SketchFiles:
    """Work with project-related files"""
    def __init__(self, project):
        self.project = project

    def path_to(self, *args):
        """Convert relative path to project"""
        self.project.assert_name()
        return os.path.join('sketches', self.project.name, *args)

    def mkdir(self, dirname):
        """Create folder inside project"""
        self.project.log('Adding folder %s to project' % self.path_to(dirname))
        os.makedirs(self.path_to(dirname), exist_ok=True)

    def open(self, *args, mode='r'):
        """Open file inside project folder"""
        self.project.log('Opening file %s in %s mode' % (self.path_to(*args), mode))
        return open(self.path_to(*args), mode=mode)

    def add(self, *args, contents, exists_ok=False):
        """Write contents to project file"""
        filename = self.path_to(*args)
        self.project.log('Adding file %s to project folder' % filename)
        self.mkdir(os.path.dirname(os.path.join(*args)))
        # prevent overwriting existing file
        if os.path.exists(filename) and not exists_ok:
            self.project.log('File already exists... skipping')
            return
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(contents)

    def cat(self, *args):
        """Print file contents to console"""
        if len(args) == 0:
            args = ['%s.ino' % self.project.name]
        self.project.log('Cat file %s' % self.path_to(*args))
        with open(self.path_to(*args), encoding="utf-8") as file:
            return file.read()
