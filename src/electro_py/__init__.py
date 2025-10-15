# Make subpackages appear under `wisco_slap.`
from . import hypno as hypno
from . import plot as plot
from . import sigpro as sigpro
from . import tdt as tdt

__all__ = ["tdt", "hypno", "sigpro", "plot"]
