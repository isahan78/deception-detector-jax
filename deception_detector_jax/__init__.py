"""DeceptionDetector-JAX: Mechanistic interpretability framework for studying deception."""

__version__ = "0.1.0"

from . import config
from . import models
from . import data
from . import interp
from . import viz
from . import evals

__all__ = [
    "config",
    "models",
    "data",
    "interp",
    "viz",
    "evals",
]
