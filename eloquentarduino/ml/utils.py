import re
import os.path
from math import pi
from collections.abc import Iterable
from jinja2 import Environment, FileSystemLoader


def prettify(code):
    """A super simple code prettifier"""
    pretty = []
    indent = 0
    for line in code.split('\n'):
        line = line.strip()
        # skip empty lines
        if len(line) == 0:
            continue
        # lower indentation on closing braces
        if line[-1] == '}' or line == '};' or line == 'protected:':
            indent -= 1
        pretty.append(('    ' * indent) + line)
        # increase indentation on opening braces
        if line[-1] == '{' or line == 'public:' or line == 'protected:':
            indent += 1
    pretty = '\n'.join(pretty)
    # leave empty line before {return, for, if}
    pretty = re.sub(r'([;])\n(\s*?)(for|return|if) ', lambda m: '%s\n\n%s%s ' % m.groups(), pretty)
    # leave empty line after closing braces
    pretty = re.sub(r'}\n', '}\n\n', pretty)
    # strip empty lines between closing braces (2 times)
    pretty = re.sub(r'\}\n\n(\s*?)\}', lambda m: '}\n%s}' % m.groups(), pretty)
    pretty = re.sub(r'\}\n\n(\s*?)\}', lambda m: '}\n%s}' % m.groups(), pretty)
    # remove "," before "}"
    pretty = re.sub(r',\s*\}', '}', pretty)
    return pretty


def jinja(template_name, template_data):
    """Render Jinja template"""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    loader = Environment(loader=FileSystemLoader(os.path.join(dir_path, "templates")))
    # custom directives
    template_data["PI"] = pi
    template_data["to_array"] = lambda arr: ", ".join([str(round(x, 9)) for x in (arr if isinstance(arr, Iterable) else [arr])])

    return prettify(loader.get_template(template_name).render(template_data))

