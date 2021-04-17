import re
import os
import os.path
import math
import numpy as np
from eloquentarduino.utils.misc import is_list
from jinja2 import Environment, FileSystemLoader, BaseLoader
from sklearn.model_selection import train_test_split


class CustomEnvironment(Environment):
    """
    Override join_path() to enable relative template paths.
    """
    def join_path(self, template, parent):
        return os.path.normpath(os.path.join(os.path.dirname(parent), template))


class CustomLoader(FileSystemLoader):
    """
    Make compatible with Windows and Unix
    """
    def get_source(self, environment, template):
        template = template.replace(os.path.sep, '/')
        return super().get_source(environment, template)


def shape(arr):
    """
    Convert array shape to C
    :param arr:
    :return:
    """
    current = arr
    shapes = []
    while is_list(current):
        shapes.append(str(len(current)))
        current = current[0]
    return ']['.join(shapes)


def to_array(arr, precision=9):
    """
    Convert array to C
    :param arr:
    :param precision:
    :return:
    """
    if not is_list(arr):
        return str(round(arr, precision))
    return '{%s}' % (', '.join([to_array(x, precision) for x in arr]))


def jinja_env(loader):
    """
    Return Jinja environment with custom options
    :param loader:
    :return:
    """
    env = CustomEnvironment(loader=loader)
    env.filters['shape'] = shape
    env.filters['to_array'] = to_array
    env.filters['ceil'] = math.ceil
    env.filters['floor'] = math.floor
    env.globals['np'] = np
    env.globals['enumerate'] = enumerate
    env.globals['isinstance'] = isinstance
    env.globals['train_test_split'] = train_test_split

    return env


def prettify(code):
    '''
    A super simple C code prettifier
    :param code: the raw C code
    :type code: str
    :return: the prettified C code
    :rtype str
    '''
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
    # remove ',' before '}'
    pretty = re.sub(r',\s*\}', '}', pretty)
    return pretty


def jinja(template_name, template_data={}, pretty=False):
    """
    Render a Jinja template

    :param template_name: the path of the template relative to the 'templates' directory
    :type template_name: str
    :param template_data: data to pass to the template, defaults to {}
    :type template_data: dict
    :param pretty: wether to prettify the code, defaults to False
    :type pretty: bool
    :return: the rendered template
    :rtype str
    """
    template_data.update(
        len=len
    )

    dir_path = os.path.dirname(os.path.realpath(__file__))
    #loader = FileSystemLoader(os.path.join(dir_path, '..', 'templates'))
    loader = CustomLoader(os.path.join(dir_path, '..', 'templates'))
    template = jinja_env(loader=loader).get_template(template_name)
    output = template.render(template_data)

    if pretty:
        output = prettify(output)

    return output


def jinja_string(template_string, template_data={}, pretty=False):
    """
    Render a Jinja template from string
    :param template_string:
    :param template_data:
    :param pretty:
    :return:
    """
    env = jinja_env(loader=BaseLoader())
    template = env.from_string(template_string)
    output = template.render(template_data)

    if pretty:
        output = prettify(output)

    return output