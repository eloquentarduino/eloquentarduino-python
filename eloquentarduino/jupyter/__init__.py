from eloquentarduino.jupyter.EloquentArduinoMagics import EloquentSketchMagics
from eloquentarduino.jupyter.project import project


def load_ipython_extension(ipython):
    """Expose extension"""
    ipython.register_magics(EloquentSketchMagics)