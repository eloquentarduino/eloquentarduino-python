from IPython.core import magic_arguments
from eloquentarduino.jupyter.project import project


class MagicMixin:
    """Utilities common to all magics"""

    @property
    def project(self):
        """Get project instance from local namespace"""
        return project

    def parse_arguments(self, method, line, local_ns=None):
        """Parse command line arguments"""
        self.local_ns = local_ns or {}
        self.local_ns.update({
            'to_array': lambda arr: ', '.join([str(x) for x in arr])
        })
        self.arguments = magic_arguments.parse_argstring(method, line)

    def log(self, *args, **kwargs):
        """Log"""
        self.project.log(*args, **kwargs)

    def path_to(self, *args):
        """Get path to given folder in current project"""
        self.project.assert_name()
        return self.project.files.path_to(*args)
