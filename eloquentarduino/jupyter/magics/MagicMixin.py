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

    def path_to(self, *args):
        """Get path to given folder in current project"""
        self.project.assert_name()
        return self.project.files.path_to(*args)
