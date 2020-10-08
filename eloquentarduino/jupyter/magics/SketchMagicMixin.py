from IPython.core import magic_arguments
from IPython.core.magic import cell_magic, needs_local_scope

from eloquentarduino.jupyter.magics.MagicMixin import MagicMixin
from eloquentarduino.utils import jinja
import re


class SketchMagicMixin(MagicMixin):
    """%%sketch magic implementation"""

    @cell_magic
    @needs_local_scope
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('filename', type=str, help='Name of the file to write')
    def sketch(self, line, code, local_ns):
        """Save code block to sketch file"""
        self.parse_arguments(self.sketch, line, local_ns)

        # if filename == main, it is the main .ino file
        if self.arguments.filename == 'main':
            self.arguments.filename = '%s.ino' % self.project.name
            self.add_eloquent_library()

        filepath = self.path_to(self.arguments.filename)
        self.log('Saving code to %s' % filepath)
        code = self.eval_python(code)
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(code)

    def add_eloquent_library(self):
        """Save the eloquent library into the sketch"""
        filepath = self.path_to('eloquent-arduino.h')
        self.log('Injecting eloquent-arduino library')
        
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(jinja('magics/eloquent-arduino.h.jinja'))

    def eval_python(self, code):
        """Interpolate Python code into sketch file"""
        # locals().update(self.local_ns or {})
        for match in re.finditer(r'\{\{\{([^{].+?)\}\}\}', code):
            source = match.group(0)
            python = match.group(1).strip()
            code = code.replace(source, str(eval(python, {}, self.local_ns)))
        return code