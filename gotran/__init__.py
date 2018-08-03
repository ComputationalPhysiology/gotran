# Copyright (C) 2012 Johan Hake
#
# This file is part of Gotran.
#
# Gotran is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Gotran is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Gotran. If not, see <http://www.gnu.org/licenses/>.

__version__ = "3.0"

import modelparameters.utils
import modelparameters.parameterdict
import modelparameters.codegeneration
import modelparameters.sympytools

# Import gotran modules
from . import common
from . import model
#import algorithms
from . import codegeneration
from . import input
from . import solver

# Import classes and routines from gotran modules
from .common import DEBUG, INFO, WARNING, ERROR, CRITICAL, \
     info, debug, warning, error, set_log_level, list_timings, \
     GotranException, parameters
from .model import *
#from algorithms import *
from .codegeneration import *
from .input import *
from .solver import *

# Model parameters
from modelparameters.parameters import ScalarParam
from modelparameters.parameterdict import ParameterDict
from modelparameters.sympytools import sp_namespace as _sp_namespace
from modelparameters.sympytools import sp as _sp

# Add sympy namespace to globals
globals().update(_sp_namespace)
globals().update(dict(eye=_sp.eye, diag=_sp.diag, Matrix=_sp.Matrix, zeros=_sp.zeros))
    
# Assign the __all__ attribute
__all__ = [name for name in list(globals().keys()) if name[:1] != "_"]
