from IPython.core.magic import Magics, magics_class

from eloquentarduino.jupyter.magics import SketchMagicMixin


@magics_class
class EloquentSketchMagics(SketchMagicMixin, Magics):
    """Define Arduino-related magics"""
    pass