from eloquentarduino.jupyter.EloquentArduinoMagics import EloquentSketchMagics
from eloquentarduino.jupyter.project import Project, project


def load_ipython_extension(ipython):
    """Expose extension"""
    ipython.register_magics(EloquentSketchMagics)