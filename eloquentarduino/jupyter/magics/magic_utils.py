import os.path
from jinja2 import FileSystemLoader, Environment


def jinja(template_name, template_data={}):
    """Render Jinja2 template"""
    templates_path = os.path.join(os.path.dirname(__file__), "templates")
    loader = Environment(loader=FileSystemLoader(templates_path))
    return loader.get_template(template_name).render(template_data)
