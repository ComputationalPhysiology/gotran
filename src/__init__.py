__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2010 " + __author__
__date__ = "2012-02-22 -- 2012-09-10"
__license__  = "GNU LGPL Version 3.0 or later"

__version__ = "2.0"

# Import gotran2 modules
import common
import model
import algorithms

# Import classes and routines from gotran2 modules
from common import DEBUG, INFO, WARNING, ERROR, CRITICAL, \
     info, debug, warning, error, set_log_level
from model import *
from algorithms import *

# Model parameters
from modelparameters.parameters import ScalarParam
from modelparameters.parameterdict import ParameterDict
from modelparameters.sympytools import sp_namespace as _sp_namespace

# Add sympy namespace to globals
globals().update(_sp_namespace)

# Assign the __all__ attribute
__all__ = [name for name in globals().keys() if name[:1] != "_"]
