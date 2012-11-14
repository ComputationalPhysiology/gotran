# Copyright (C) 2011-2012 Johan Hake
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

__all__ = ["ODEObject", "Parameter", "State", "Variable"]#
#           "DiscreteVariable", "ContinuousVariable", ]

# System imports
import numpy as np

# ModelParameters imports
from modelparameters.sympytools import sp, ModelSymbol
from modelparameters.parameters import *

from gotran.common import error, check_arg, scalars

class ODEObject(object):
    """
    Base container class for all ODEObjects
    """
    def __init__(self, name, init, comment="", ode_name=""):
        """
        Create ODEObject instance

        Arguments
        ---------
        name : str
            The name of the ODEObject
        init : scalar, ScalarParam
            The initial value of this ODEObject
        comment : str (optional)
            A comment about the ODEObject
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """

        check_arg(name, str, 0, ODEObject)
        check_arg(init, scalars + (ScalarParam, list, np.ndarray), \
                  1, ODEObject)
        
        if isinstance(init, ScalarParam):

            # If Param already has a symbol
            if init.sym != dummy_sym:

                # Re create one without a name
                init = eval(repr(init).split(", name")[0]+")")

        elif isinstance(init, scalars):
            init = ScalarParam(init)
        else:
            init = ArrayParam(np.fromiter(init, dtype=np.float_))
            
        # Create a symname based on the name of the ODE
        if ode_name:
            init.name = name, "{0}.{1}".format(ode_name, name)
        else:
            init.name = name

        # Store the Param
        self._param = init 

        # Store field
        # FIXME: Is this nesesary
        self._field = isinstance(init, ArrayParam)
        self._comment = comment
        self._ode_name = ode_name

    @property
    def is_field(self):
        return self._field

    @property
    def sym(self):
        return self._param.sym

    @property
    def name(self):
        return self._param.name

    @property
    def param(self):
        return self._param

    @property
    def comment(self):
        return self._comment

    @property
    def init(self):
        return self._param.getvalue()

    @init.setter
    def init(self, value):
        self._param.setvalue(value)

    def __eq__(self, other):
        """
        x.__eq__(y) <==> x==y
        """
        check_arg(other, (str, ODEObject, ModelSymbol))
        return self.name == str(other)

    def __str__(self):
        """
        x.__str__() <==> str(x)
        """
        return self.name

    def __repr__(self):
        """
        x.__repr__() <==> repr(x)
        """
        return "{0}({1})".format(self.__class__.__name__, self._args_str())

    def _args_str(self):
        """
        Return a formated str of __init__ arguments
        """
        return "'{0}', {1}{2}{3}".format(\
            self.name, repr(self.init),
            ", comment='{0}'".format(self._comment) \
            if self._comment else "",
            ", ode_name='{0}'".format(self._ode_name) \
            if self._ode_name else "",)

class State(ODEObject):
    """
    Container class for a State variable
    """
    def __init__(self, name, init, comment="", ode_name=""):
        """
        Create a state variable with an assosciated initial value

        Arguments
        ---------
        name : str
            The name of the State
        init : scalar, ScalarParam
            The initial value of this state
        comment : str (optional)
            A comment about the State
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """
        
        # Call super class
        super(State, self).__init__(name, init, comment, ode_name)

        # Add an attribute to register dependencies
        self.dependencies = []
        self.linear_dependencies = []

        # Add previous value symbol
        if ode_name:
            self.sym_0 = ModelSymbol("{0}_0".format(name), \
                                     "{0}.{1}_0".format(ode_name, name))
        else:
            self.sym_0 = ModelSymbol("{0}_0".format(name))
    
class Parameter(ODEObject):
    """
    Container class for a Parameter
    """
    def __init__(self, name, init, comment="", ode_name=""):
        """
        Create a Parameter with an assosciated initial value

        Arguments
        ---------
        name : str
            The name of the State
        init : scalar, ScalarParam
            The initial value of this parameter
        comment : str (optional)
            A comment about the Parameter
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """
        
        # Call super class
        super(Parameter, self).__init__(name, init, comment, ode_name)

class Variable(ODEObject):
    """
    Container class for a Variable
    """
    def __init__(self, name, init, comment="", ode_name=""):
        """
        Create a variable with an assosciated initial value

        Arguments
        ---------
        name : str
            The name of the variable
        init : scalar
            The initial value of this variable
        comment : str (optional)
            A comment about the Variables
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """
        
        # Call super class
        super(Variable, self).__init__(name, init, comment, ode_name)

        # Add previous value symbol
        self.sym_0 = ModelSymbol("{0}_0".format(name), \
                                 "{0}.{1}_0".format(ode_name, name))


