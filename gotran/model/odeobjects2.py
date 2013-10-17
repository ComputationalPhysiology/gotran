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

__all__ = ["ODEObject", "ODEValueObject", "Parameter", "State", \
           "StateDerivative", "Variable", "SingleODEObjects", "Time", "Dt"]

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
    def __init__(self, name):
        """
        Create ODEObject instance

        Arguments
        ---------
        name : str
            The name of the ODEObject
        """

        check_arg(name, str, 0, ODEObject)
        self._name = self._check_name(name)

    def __eq__(self, other):
        """
        x.__eq__(y) <==> x==y
        """
        
        if not isinstance(other, type(self)):
            return False
        
        return self._name == str(other)

    def __ne__(self, other):
        """
        x.__neq__(y) <==> x==y
        """
        
        if not isinstance(other, type(self)):
            return True
        
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
        return "'{0}'".format(self._name)
    
    def rename(self, name):
        """
        Rename the ODEObject
        """

        check_arg(name, str)
        self._name = self._check_name(name)

    def _check_name(self, name):
        """
        Check the name
        """
        assert(isinstance(name, str))
        name = name.strip().replace(" ", "_")

        # Check for underscore in name
        if len(name) > 0 and name[0] == "_":
            error("No ODEObject names can start with an underscore: "\
                  "'{0}'".format(name))
            
        return name

class ODEValueObject(ODEObject):
    """
    A class for all ODE objects which has a value
    """
    def __init__(self, name, value):
        """
        Create ODEObject instance

        Arguments
        ---------
        name : str
            The name of the ODEObject
        value : scalar, ScalarParam, np.ndarray, sp. Basic
            The value of this ODEObject
        """

        check_arg(name, str, 0, ODEValueObject)
        check_arg(value, scalars + (ScalarParam, list, np.ndarray, sp.Basic), \
                  1, ODEValueObject)

        # Init super class
        super(ODEValueObject, self).__init__(name)

        if isinstance(value, ScalarParam):

            # Re-create one with correct name
            value = value.copy(include_name=False)
            value.name = name
            
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

    def rename(self, name):
        """
        Rename the ODEValueObject
        """
        super(ODEValueObject, self).rename(name)

        # Re-create param with changed name
        param = self._param.copy(include_name=False)
        param.name = name
        self._param = param
        
    @property
    def value(self):
        return self._param.getvalue()

    @value.setter
    def value(self, value):
        self._param.setvalue(value)

    @property
    def is_field(self):
        return isinstance(self._param, ArrayParam)

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
            self.name, repr(self._param.getvalue()))

class State(ODEValueObject):
    """
    Container class for a State variable
    """
    def __init__(self, name, init, time):
        """
        Create a state variable with an assosciated initial value

        Arguments
        ---------
        name : str
            The name of the State
        init : scalar, ScalarParam
            The initial value of this state
        time : Variable
            The time variable
        """
        
        # Call super class
        check_arg(init, scalars + (ScalarParam, list, np.ndarray), \
                  1, State)
        
        super(State, self).__init__(name, init)
        check_arg(time, Time, 2)

        self.time = time

        self.derivative = StateDerivative(self)

        # Add previous value symbol
        self.sym_0 = sp.Symbol("{0}_0".format(name))(time.sym)

    init = ODEValueObject.value

    @property
    def sym(self):
        return self._param.sym(self.time.sym)

    def _args_str(self):
        """
        Return a formated str of __init__ arguments
        """
        return "'{0}', {1}, {2}".format(\
            self.name, repr(self._param.copy(include_name=False)), self.time)

    def toggle_field(self):
        """
        Toggle a State between scalar and field object
        """
        if isinstance(self._param, ArrayParam):
            self._param = eval("ScalarParam(%s%s%s)" % \
                               (self._param.value[0], \
                                self._param._check_arg(), \
                                self._param._name_arg()))
        else:
            self._param = eval("ArrayParam(%s, 1%s%s)" % \
                               (self._param.value, \
                                self._param._check_arg(), \
                                self._param._name_arg()))
            
class StateDerivative(ODEValueObject):
    """
    Container class for a StateDerivative variable
    """
    def __init__(self, state, init=0.0):
        """
        Create a state derivative variable with an assosciated initial value

        Arguments
        ---------
        state : State
            The State
        init : scalar, ScalarParam
            The initial value of this state derivative
        """

        check_arg(state, State)

        # Call super class
        super(StateDerivative, self).__init__("d{0}_dt".format(state.name), \
                                              init)

        self.time = state.time
        self.state = state
    
    init = ODEValueObject.value

    @property
    def sym(self):
        return self.state.sym.diff(self.time.sym)

class Parameter(ODEValueObject):
    """
    Container class for a Parameter
    """
    def __init__(self, name, init):
        """
        Create a Parameter with an assosciated initial value

        Arguments
        ---------
        name : str
            The name of the State
        init : scalar, ScalarParam
            The initial value of this parameter
        """
        
        # Call super class
        super(Parameter, self).__init__(name, init)
    
    init = ODEValueObject.value

    def toggle_field(self):
        """
        Toggle a State between scalar and field object
        """
        if isinstance(self._param, ArrayParam):
            self._param = eval("ScalarParam(%s%s%s)" % \
                               (self._param.value[0], \
                                self._param._check_arg(), \
                                self._param._name_arg()))
        else:
            self._param = eval("ArrayParam(%s, 1%s%s)" % \
                               (self._param.value, \
                                self._param._check_arg(), \
                                self._param._name_arg()))
    
class Variable(ODEValueObject):
    """
    Container class for a Variable
    """
    def __init__(self, name, init):
        """
        Create a variable with an assosciated initial value

        Arguments
        ---------
        name : str
            The name of the variable
        init : scalar
            The initial value of this variable
        """
        
        # Call super class
        super(Variable, self).__init__(name, init)

        # Add previous value symbol
        self.sym_0 = sp.Symbol("{0}_0".format(name))

    init = ODEValueObject.value

class Time(Variable):
    """
    Specialization for a Time class
    """
    def __init__(self, name):
        super(Time, self).__init__(name, 0.0)

class Dt(Variable):
    """
    Specialization for a time step class
    """
    def __init__(self, name):
        super(Dt, self).__init__(name, 0.1)
    
# Tuple with single ODE Objects, for type checking
SingleODEObjects = (State, StateDerivative, Parameter, Variable, Time)
