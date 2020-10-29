import re
import os
import os.path
from eloquentarduino.utils.misc import is_list
from jinja2 import Environment, FileSystemLoader, BaseLoader


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
    env = Environment(loader=loader)
    env.filters['shape'] = shape
    env.filters['to_array'] = to_array

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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    loader = Environment(loader=FileSystemLoader(os.path.join(dir_path, '..', 'templates')))
    # custom directives
    template_data['len'] = len
    template_data['enumerate'] = enumerate
    template_data['to_array'] = lambda arr: ', '.join([str(round(x, 9)) for x in (arr if isinstance(arr, Iterable) else [arr])])

    output = loader.get_template(template_name).render(template_data)

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
    output = jinja_env(loader=BaseLoader()).from_string(template_string).render(template_data)

    if pretty:
        output = prettify(output)

    return output