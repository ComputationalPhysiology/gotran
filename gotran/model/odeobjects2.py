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

__all__ = ["ODEObject", "Comment", "ODEValueObject", "Parameter", "State", \
           "SingleODEObjects", "Time", "Dt"]

# System imports
import numpy as np
from collections import OrderedDict
from sympy.core.function import AppliedUndef

# ModelParameters imports
from modelparameters.sympytools import sp, symbols_from_expr
from modelparameters.codegeneration import sympycode, latex, latex_unit
from modelparameters.parameters import *

from gotran.common import error, check_arg, scalars, debug, DEBUG, \
     get_log_level, Timer

class ODEObject(object):
    """
    Base container class for all ODEObjects
    """
    __count = 0
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

        # Unique identifyer
        self._hash = ODEObject.__count
        ODEObject.__count += 1

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        """
        x.__eq__(y) <==> x==y
        """

        if not isinstance(other, type(self)):
            return False

        return self._hash == other._hash

    def __ne__(self, other):
        """
        x.__neq__(y) <==> x==y
        """

        if not isinstance(other, type(self)):
            return True

        return self._hash != other._hash

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
        Return a formatted str of __init__ arguments
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

        # Check for underscore in name
        if len(name) > 0 and name[0] == "_":
            error("No ODEObject names can start with an underscore: "\
                  "'{0}'".format(name))

        return name

class Comment(ODEObject):
    """
    A Comment. To keep track of user comments in an ODE
    """
    def __init__(self, comment):
        """
        Create a comment

        Arguments
        ---------
        comment : str
            The comment
        """

        # Call super class
        super(Comment, self).__init__(comment)

    def _repr_latex_(self):
        return "$\\mathbf{{{0}}}$".format(self.name.replace(" ", "\\;"))

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
            value = ArrayParam(value, name=name)

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
        self._arg_ind = -1

    def rename(self, name):
        """
        Rename the ODEValueObject
        """
        super(ODEValueObject, self).rename(name)

        # Re-create param with changed name
        param = self._param.copy(include_name=False)
        param.name = name
        self._param = param

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
        Return a formatted str of __init__ arguments
        """
        return "'{0}', {1}".format(self.name, self._param.repr(\
            include_name=False))

    def _repr_latex_(self):
        """
        Return a pretty latex representation of the ODEValue object
        """
        value = self.value[0] if self.is_field else self.value
        unit_str = latex_unit(self.param.unit)
        return "${0}{1}$".format(latex(value), "\\;{0}".format(unit_str) \
                                 if unit_str else "")

    @property
    def arg_ind(self):
        """
        Attribute to determine an argument index
        """
        return self._arg_ind

    @arg_ind.setter
    def arg_ind(self, value):
        """
        Set argument index
        """
        value = tuplewrap(value)
        check_arg(value, tuple, itemtype=int)
        self_arg_ind = value

class State(ODEValueObject):
    """
    class for a State variable
    """
    def __init__(self, name, init, time):
        """
        Create a state variable with an associated initial value

        Arguments
        ---------
        name : str
            The name of the State
        init : scalar, ScalarParam
            The initial value of this state
        time : Time
            The time variable
        """

        # Call super class
        check_arg(init, scalars + (ScalarParam, list, np.ndarray), \
                  1, State)

        super(State, self).__init__(name, init)
        check_arg(time, Time, 2)

        self.time = time

        # Add previous value symbol
        self.sym_0 = sp.Symbol("{0}_0".format(name))(time.sym)

        # Flag to determine if State is solved or not
        self._is_solved = False

    init = ODEValueObject.value

    @property
    def sym(self):
        return self._param.sym(self.time.sym)

    def _args_str(self):
        """
        Return a formatted str of __init__ arguments
        """
        return "'{0}', {1}, {2}".format(\
            self.name, repr(self._param.copy(include_name=False)), repr(self.time))

    def toggle_field(self):
        """
        Toggle a State between scalar and field object
        """
        if isinstance(self._param, ArrayParam):
            self._param = eval("ScalarParam(%s%s%s)" % \
                               (self._param.value[0], \
                                self._param._check_arg(), \
                                self._param._name_arg()))
        elif not self.is_solved:
            
            self._param = eval("ArrayParam(%s, 1%s%s)" % \
                               (self._param.value, \
                                self._param._check_arg(), \
                                self._param._name_arg()))
        else:
            error("Cannot turn a solved state into a field state")

    @property
    def is_solved(self):
        return self._is_solved

class Parameter(ODEValueObject):
    """
    class for a Parameter
    """
    def __init__(self, name, init):
        """
        Create a Parameter with an associated initial value

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

class Argument(ODEValueObject):
    """
    class for an arbitrary argument used to add arguments to CodeComponents
    """
    def __init__(self, name):
        """
        Create a Parameter with an associated initial value

        Arguments
        ---------
        name : str
            The name of the argument
        """

        # Call super class
        super(Argument, self).__init__(name, 0.0)

class Time(ODEValueObject):
    """
    Specialization for a Time class
    """
    def __init__(self, name, unit="ms"):
        super(Time, self).__init__(name, ScalarParam(0.0, unit=unit))

        # Add previous value symbol
        self.sym_0 = sp.Symbol("{0}_0".format(name))

class Dt(ODEValueObject):
    """
    Specialization for a time step class
    """
    def __init__(self, name):
        super(Dt, self).__init__(name, 0.1)


# Tuple with single ODE Objects, for type checking
SingleODEObjects = (State, Parameter, Time)
