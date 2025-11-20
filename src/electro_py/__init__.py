# Make subpackages appear under `wisco_slap.`
from . import hypno as hypno
from . import plot as plot
from . import sigpro as sigpro
from . import tdt as tdt
from . import util as util
from . import gen as gen

__all__ = ["tdt", "hypno", "sigpro", "plot", "util", "gen"]
