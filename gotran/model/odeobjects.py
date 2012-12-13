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

__all__ = ["ODEObject", "SingleODEObject", "Parameter", "State", \
           "StateDerivative", "Variable"]

# System imports
import numpy as np
from collections import OrderedDict

# ModelParameters imports
from modelparameters.sympytools import sp, ModelSymbol, \
     iter_symbol_params_from_expr
from modelparameters.parameters import *

from gotran.common import error, check_arg, scalars, debug, DEBUG, \
     get_log_level, Timer

class ODEObject(object):
    """
    Base container class for all ODEObjects
    """
    def __init__(self, name, value, component="", ode_name=""):
        """
        Create ODEObject instance

        Arguments
        ---------
        name : str
            The name of the ODEObject
        value : scalar, ScalarParam, np.ndarray, sp. Basic, str
            The value of this ODEObject
        component : str (optional)
            A component about the ODEObject
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """

        check_arg(name, str, 0, ODEObject)
        check_arg(value, scalars + (ScalarParam, list, np.ndarray, sp.Basic, str), \
                  1, ODEObject)
        check_arg(component, str, 2, ODEObject)
        check_arg(ode_name, str, 3, ODEObject)

        # Check for underscore in name
        if len(name) > 0 and name[0] == "_":
            error("No ODEObject names can start with an underscore: "\
                  "'{0}'".format(name))

        # Strip name for spaces
        _name = name.strip().replace(" ", "_")

        if isinstance(value, ScalarParam):

            # If Param already has a symbol
            if value.sym != dummy_sym:

                # Re-create one without a name
                value = eval(repr(value).split(", name")[0]+")")

        elif isinstance(value, scalars):
            value = ScalarParam(value)
        
        elif isinstance(value, (list, np.ndarray)):
            value = ArrayParam(np.fromiter(value, dtype=np.float_))
        elif isinstance(value, str):
            value = ConstParam(value)
        else:
            value = SlaveParam(value)

        # Debug
        if get_log_level() <= DEBUG:
            if isinstance(value, SlaveParam):
                debug("{0}: {1} {2:.3f}".format(name, value.expr, value.value))
            else:
                debug("{0}: {1}".format(name, value.value))
            
        # Create a symname based on the name of the ODE
        if ode_name:
            value.name = name, "{0}.{1}".format(ode_name, name)
        else:
            value.name = name

        # Store the Param
        self._param = value 

        # Store field
        self._field = isinstance(value, ArrayParam)
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
        return self.name == str(other)

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
        return self.name != str(other)

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
            self.name, repr(self._param.getvalue()),
            ", component='{0}'".format(self._component) \
            if self._component else "",
            ", ode_name='{0}'".format(self._ode_name) \
            if self._ode_name else "",)

class SingleODEObject(ODEObject):
    """
    A class for all ODE objects which are not compound
    """
    
    def __init__(self, name, init, component="", ode_name=""):
        """
        Create ODEObject instance

        Arguments
        ---------
        name : str
            The name of the ODEObject
        init : scalar, ScalarParam, np.ndarray
            The init value of this ODEObject
        component : str (optional)
            A component about the ODEObject
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """

        # Init super class
        super(SingleODEObject, self).__init__(name, init, component, ode_name)

    @property
    def init(self):
        return self._param.getvalue()

    @init.setter
    def init(self, value):
        self._param.setvalue(value)

class State(SingleODEObject):
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

        self.derivative = None

        # Add previous value symbol
        if ode_name:
            self.sym_0 = ModelSymbol("{0}_0".format(name), \
                                     "{0}.{1}_0".format(ode_name, name))
        else:
            self.sym_0 = ModelSymbol("{0}_0".format(name))
    
class StateDerivative(SingleODEObject):
    """
    Container class for a StateDerivative variable
    """
    def __init__(self, state, init=0.0, component="", ode_name=""):
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
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """

        check_arg(state, State)

        # Call super class
        super(StateDerivative, self).__init__("d{0}_dt".format(state.name), \
                                              init, component, ode_name)
        
        self.state = state
    
class Parameter(SingleODEObject):
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

class Variable(SingleODEObject):
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

