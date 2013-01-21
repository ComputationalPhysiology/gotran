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

__version__ = "2.0"

# Import gotran modules
import common
import model
import algorithms
import codegeneration

# Import classes and routines from gotran modules
from common import DEBUG, INFO, WARNING, ERROR, CRITICAL, \
     info, debug, warning, error, set_log_level, list_timings
from model import *
from algorithms import *
from codegeneration import jit

# Model parameters
from modelparameters.parameters import ScalarParam
from modelparameters.parameterdict import ParameterDict
from modelparameters.sympytools import sp_namespace as _sp_namespace

# Add sympy namespace to globals
globals().update(_sp_namespace)

# Assign the __all__ attribute
__all__ = [name for name in globals().keys() if name[:1] != "_"]
