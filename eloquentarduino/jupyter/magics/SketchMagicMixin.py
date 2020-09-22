from IPython.core import magic_arguments
from IPython.core.magic import cell_magic, needs_local_scope

from eloquentarduino.jupyter.magics.MagicMixin import MagicMixin
from eloquentarduino.jupyter.magics.magic_utils import jinja


class SketchMagicMixin(MagicMixin):
    """%%sketch magic implementation"""

    @cell_magic
    @needs_local_scope
    @magic_arguments.magic_arguments()
    @magic_arguments.argument("filename", type=str, help="Name of the file to write")
    def sketch(self, line, code, local_ns):
        """Save code block to sketch file"""
        self.parse_arguments(self.sketch, line, local_ns)

        # if filename == main, it is the main .ino file
        if self.arguments.filename == "main":
            self.arguments.filename = "%s.ino" % self.project.name

        filepath = self.path_to(self.arguments.filename)
        self.log("Saving code to %s", filepath)
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(self._inject_eloquent_library(code))

    def _inject_eloquent_library(self, code):
        """Add the eloquent library to the sketch"""
        if '%no eloquent' in code:
            return code.replace('%no eloquent', '')
        self.log("Injecting eloquent-arduino library")
        filepath = self.path_to("eloquent-arduino.h")
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(jinja("eloquent-arduino.h.jinja"))
        return '#include "eloquent-arduino.h"\n' + code