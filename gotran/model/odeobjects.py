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
from collections import deque

# ModelParameters imports
from modelparameters.sympytools import sp, ModelSymbol
from modelparameters.parameters import *

from gotran.common import error, check_arg, scalars

class ODEObject(object):
    """
    Base container class for all ODEObjects
    """
    def __init__(self, name, init, component="", ode_name=""):
        """
        Create ODEObject instance

        Arguments
        ---------
        name : str
            The name of the ODEObject
        init : scalar, ScalarParam
            The initial value of this ODEObject
        component : str (optional)
            A component about the ODEObject
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """

        check_arg(name, str, 0, ODEObject)
        check_arg(init, scalars + (ScalarParam, list, np.ndarray, sp.Basic), \
                  1, ODEObject)
        
        if isinstance(init, ScalarParam):

            # If Param already has a symbol
            if init.sym != dummy_sym:

                # Re create one without a name
                init = eval(repr(init).split(", name")[0]+")")

        elif isinstance(init, scalars):
            init = ScalarParam(init)
        elif isinstance(init, (list, np.ndarray)):
            init = ArrayParam(np.fromiter(init, dtype=np.float_))
        else:
            init = SlaveParam(init)
            
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
        self._component = component
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
    def component(self):
        return self._component

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
            ", component='{0}'".format(self._component) \
            if self._component else "",
            ", ode_name='{0}'".format(self._ode_name) \
            if self._ode_name else "",)

class State(ODEObject):
    """
    Container class for a State variable
    """
    def __init__(self, name, init, component="", ode_name=""):
        """
        Create a state variable with an assosciated initial value

        Arguments
        ---------
        name : str
            The name of the State
        init : scalar, ScalarParam
            The initial value of this state
        component : str (optional)
            A component for which the State should be associated with.
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """
        
        # Call super class
        super(State, self).__init__(name, init, component, ode_name)

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
    def __init__(self, name, init, component="", ode_name=""):
        """
        Create a Parameter with an assosciated initial value

        Arguments
        ---------
        name : str
            The name of the State
        init : scalar, ScalarParam
            The initial value of this parameter
        component : str (optional)
            A component for which the Parameter should be associated with.
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """
        
        # Call super class
        super(Parameter, self).__init__(name, init, component, ode_name)

class Variable(ODEObject):
    """
    Container class for a Variable
    """
    def __init__(self, name, init, component="", ode_name=""):
        """
        Create a variable with an assosciated initial value

        Arguments
        ---------
        name : str
            The name of the variable
        init : scalar
            The initial value of this variable
        component : str (optional)
            A component for which the Variable should be associated with.
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """
        
        # Call super class
        super(Variable, self).__init__(name, init, component, ode_name)

        # Add previous value symbol
        self.sym_0 = ModelSymbol("{0}_0".format(name), \
                                 "{0}.{1}_0".format(ode_name, name))

class Expression(ODEObject):
    """
    class for all expressions such as intermediates and diff 
    """
    def __init__(self, name, expr, expanded_expr, component="", ode_name=""):
        """
        Create an Exression with an assosciated name

        Arguments
        ---------
        name : str
            The name of the Expression
        expr : sympy.Basic
            The expression 
        expanded_expr : sympy.Basic
            The expanded verision of the expression 
        component : str (optional)
            A component for which the Expression should be associated with.
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """

        # Check arguments
        check_arg(expr, sp.Basic)
        if not any(isinstance(atom, ModelSymbol) for atom in expr.atoms()):
            error("expected the expression to contain at least one ModelSymbol.")

        check_arg(expanded_expr, sp.Basic)
        if not any(isinstance(atom, ModelSymbol) \
                   for atom in expanded_expr.atoms()):
            error("expected the expanded_expr to contain at least "\
                  "one ModelSymbol.")
        
        # Call super class with expression as the "init" value
        super(Expression, self).__init__(name, expr, component, ode_name)

        # Store the expanded expression
        self._expanded_expr = expanded_expr

    @property
    def expr(self):
        """
        Return the stored expression
        """
        return self._param.expr

    @property
    def expanded_expr(self):
        """
        Return the stored expression
        """
        return self._expanded_expr
    
