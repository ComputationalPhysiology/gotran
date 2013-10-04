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

__all__ = ["ODEObject", "ValueODEObject", "Parameter", "State", \
           "StateDerivative", "Variable", "SingleODEObjects"]

# System imports
import numpy as np
from collections import OrderedDict

# ModelParameters imports
from modelparameters.sympytools import sp, iter_symbol_params_from_expr
from modelparameters.parameters import *

from gotran.common import error, check_arg, scalars, debug, DEBUG, \
     get_log_level, Timer

class ODEObject(object):
    """
    Base container class for all ODEObjects
    """
    def __init__(self, name, component=""):
        """
        Create ODEObject instance

        Arguments
        ---------
        name : str
            The name of the ODEObject
        component : str (optional)
            A component about the ODEObject
        """

        check_arg(name, str, 0, ODEObject)
        check_arg(component, str, 1, ODEObject)

        # Check for underscore in name
        if len(name) > 0 and name[0] == "_":
            error("No ODEObject names can start with an underscore: "\
                  "'{0}'".format(name))

        self._name = name
        self._component = component

    @property
    def component(self):
        return self._component

    def __eq__(self, other):
        """
        x.__eq__(y) <==> x==y
        """
        
        if not isinstance(other, type(self)):
            return False
        
        # FIXME: Should this be more restrictive? Only comparing ODEObjects,
        # FIXME: and then comparing name and component?
        # FIXME: Yes, might be some side effects though...
        # FIXME: Need to do change when things are stable
        return self._name == str(other)

    def __ne__(self, other):
        """
        x.__neq__(y) <==> x==y
        """
        
        if not isinstance(other, type(self)):
            return True
        
        # FIXME: Should this be more restrictive? Only comparing ODEObjects,
        # FIXME: and then comparing name and component?
        # FIXME: Yes, might be some side effects though...
        # FIXME: Need to do change when things are stable
        return self._name != str(other)

    def __str__(self):
        """
        x.__str__() <==> str(x)
        """
        return self._name

    def __repr__(self):
        """
        x.__repr__() <==> repr(x)
        """
        return "{0}({1})".format(self.__class__.__name__, self._args_str())

    @property
    def name(self):
        return self._name

    def _args_str(self):
        """
        Return a formated str of __init__ arguments
        """
        return "'{0}', {1}{2}".format(self._name,\
            ", component='{0}'".format(self._component) \
            if self._component else "")

class ValueODEObject(ODEObject):
    """
    A class for all ODE objects which has a value
    """
    def __init__(self, name, value, component="", slaved=False):
        """
        Create ODEObject instance

        Arguments
        ---------
        name : str
            The name of the ODEObject
        value : scalar, ScalarParam, np.ndarray, sp. Basic
            The value of this ODEObject
        component : str (optional)
            A component about the ODEObject
        slaved : bool
            If True the creation and differentiation is controlled by
            other entity, like a Markov model.
        """

        check_arg(name, str, 0, ValueODEObject)
        check_arg(value, scalars + (ScalarParam, list, np.ndarray, sp.Basic), \
                  1, ValueODEObject)

        name = name.strip().replace(" ", "_")
        
        # Init super class
        super(ValueODEObject, self).__init__(name, component)

        if isinstance(value, ScalarParam):

            # Re-create one with correct name
            if value.name:
                value = eval(repr(value).split(", name")[0]+", name=name)")
            else:
                value = eval(repr(value).replace(")", ", name=name)"))

        elif isinstance(value, scalars):
            value = ScalarParam(value, name=name)
        
        elif isinstance(value, (list, np.ndarray)):
            value = ArrayParam(np.fromiter(value, dtype=np.float_), name=name)
        elif isinstance(value, str):
            value = ConstParam(value, name=name)
        else:
            value = SlaveParam(value, name=name)

        # Debug
        if get_log_level() <= DEBUG:
            if isinstance(value, SlaveParam):
                debug("{0}: {1} {2:.3f}".format(self.name, value.expr, value.value))
            else:
                debug("{0}: {1}".format(name, value.value))
            
        # Store the Param
        self._param = value 

        # Store field
        self._field = isinstance(value, ArrayParam)

        self._slaved = slaved

    @property
    def slaved(self):
        return self._slaved
    
    @property
    def value(self):
        return self._param.getvalue()

    @value.setter
    def value(self, value):
        self._param.setvalue(value)


    @property
    def is_field(self):
        return self._field

    @property
    def sym(self):
        return self._param.sym

    @property
    def param(self):
        return self._param

    def _args_str(self):
        """
        Return a formated str of __init__ arguments
        """
        return "'{0}', {1}{2}".format(\
            self.name, repr(self._param.getvalue()),
            ", component='{0}'".format(self._component) \
            if self._component else "")

class State(ValueODEObject):
    """
    Container class for a State variable
    """
    def __init__(self, name, init, component="", slaved=False):
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
        slaved : bool
            If True the creation and differentiation is controlled by
            other entity, like a Markov model.
        """
        
        # Call super class
        super(State, self).__init__(name, init, component, slaved)

        self.derivative = None

        check_arg(slaved, bool, 4)
        self._slaved = slaved

        # Add previous value symbol
        self.sym_0 = sp.Symbol("{0}_0".format(name))

    init = ValueODEObject.value

class StateDerivative(ValueODEObject):
    """
    Container class for a StateDerivative variable
    """
    def __init__(self, state, init=0.0, component=""):
        """
        Create a state derivative variable with an assosciated initial value

        Arguments
        ---------
        state : State
            The State
        init : scalar, ScalarParam
            The initial value of this state derivative
        component : str (optional)
            A component for which the State should be associated with.
        """

        check_arg(state, State)

        # Call super class
        super(StateDerivative, self).__init__("d{0}_dt".format(state.name), \
                                              init, component)
        
        self.state = state
    
    init = ValueODEObject.value

class Parameter(ValueODEObject):
    """
    Container class for a Parameter
    """
    def __init__(self, name, init, component=""):
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
        """
        
        # Call super class
        super(Parameter, self).__init__(name, init, component)
    
    init = ValueODEObject.value

class Variable(ValueODEObject):
    """
    Container class for a Variable
    """
    def __init__(self, name, init, component=""):
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
        """
        
        # Call super class
        super(Variable, self).__init__(name, init, component)

        # Add previous value symbol
        self.sym_0 = sp.Symbol("{0}_0".format(name))

    init = ValueODEObject.value

# Tuple with single ODE Objects, for type checking
SingleODEObjects = (State, StateDerivative, Parameter, Variable)
