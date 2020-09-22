import os.path

from IPython.core import magic_arguments


class MagicMixin:
    """Utilities common to all magics"""
    def parse_arguments(self, method, line, local_ns=None):
        """Parse command line arguments"""
        self.arguments = magic_arguments.parse_argstring(method, line)
        if local_ns is not None:
            self.project = local_ns.get('project', None)

    def log(self, *args, **kwargs):
        """Log"""
        self.project.log(*args, **kwargs)

    def path_to(self, folder=""):
        """Get path to given folder in current project"""
        assert self.project, "You MUST create a project first"
        assert self.project.name, "You MUST set a project name"
        return os.path.join("sketches", self.project.name, folder)