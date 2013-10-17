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

__all__ = ["ODE"]

# System imports
import inspect
import re
from collections import OrderedDict, deque

# ModelParameter imports
from modelparameters.sympytools import sp, sp_namespace
from modelparameters.codegeneration import sympycode, _all_keywords
from modelparameters.parameters import ScalarParam
from modelparameters.sympytools import iter_symbol_params_from_expr

# Gotran imports
from gotran.common import type_error, value_error, error, check_arg, \
     check_kwarg, scalars, listwrap, info, debug, Timer
from gotran.model.odeobjects2 import *
from gotran.model.odecomponents2 import *
from gotran.model.expressions2 import *

_derivative_name_template = re.compile("\Ad([a-zA-Z]\w*)_dt\Z")

class ODE(object):
    """
    Basic class for storying information of an ODE
    """
    def __init__(self, name, return_namespace=False):
        """
        Initialize an ODE
        
        Arguments
        ---------
        name : str
            The name of the ODE
        return_namespace : bool (optional)
            Several methods will return a namespace dict populated with symbols
            added to the ODE if return_namespace is set to True.
        """
        check_arg(name, str, 0)

        self._return_namespace = return_namespace

        # Initialize attributes
        self._name = name.strip().replace(" ", "_")

        self.objects = dict()
        self.components = OrderedDict()
        self.components[name] = self._default_component

    def signature(self):
        """
        Return a signature uniquely defining the ODE
        """
        import hashlib
        def_list = [repr(state.param) for state in self.states] 
        def_list += [repr(param.param) for param in self.parameters] 
        def_list += [repr(variable.param) for variable in self.variables] 
        def_list += [str(intermediate.expr) if isinstance(intermediate, Expression) \
                     else str(intermediate) for intermediate in self.intermediates]
        def_list += [str(monitor) for monitor in self.monitored_intermediates]
        def_list += [str(expr) for expr in self.get_derivative_expr()]

        h = hashlib.sha1()
        h.update(";".join(def_list))
        return h.hexdigest()
        
