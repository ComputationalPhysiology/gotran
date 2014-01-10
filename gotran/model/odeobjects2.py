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
           "SingleODEObjects", "Time", "Dt", "IndexedObject"]

# System imports
import numpy as np
from collections import OrderedDict
from sympy.core.function import AppliedUndef

# ModelParameters imports
from modelparameters.sympytools import sp, symbols_from_expr
from modelparameters.codegeneration import sympycode, latex, latex_unit
from modelparameters.parameters import *

from gotran.common import error, check_arg, scalars, debug, DEBUG, \
     get_log_level, Timer, parameters

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
        self._count = ODEObject.__count
        ODEObject.__count += 1

    def __hash__(self):
        return id(self)

    def __cmp__(self, other):
        if not isinstance(other, ODEObject):
            return -1
        return cmp(self._count, other._count)

    def __repr__(self):
        """
        x.__repr__() <==> repr(x)
        """
        return "{0}({1})".format(self.__class__.__name__, self._args_str())

    def __str__(self):
        """
        x.__str__() <==> str(x)
        """
        return self.name

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

    def _recount(self, new_count=None):
        """
        Method called when an object need to get a new count
        """
        old_count = self._count
        if new_count is None:
            self._count = ODEObject.__count
            ODEObject.__count += 1
        else:
            check_arg(new_count, int, ge=0, le=ODEObject.__count)
            self._count = new_count
            
        debug("Change count of {0} from {1} to {2}".format(\
            self.name, old_count, self._count))
    
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
        check_arg(value, scalars + (ScalarParam, sp.Basic), 1, ODEValueObject)

        # Init super class
        super(ODEValueObject, self).__init__(name)

        if isinstance(value, ScalarParam):

            # Re-create one with correct name
            value = value.copy(include_name=False)
            value.name = name

        elif isinstance(value, scalars):
            value = ScalarParam(value, name=name)

        elif isinstance(value, str):
            value = ConstParam(value, name=name)

        else:
            value = SlaveParam(value, name=name)

        # Debug
        #if get_log_level() <= DEBUG:
        #    if isinstance(value, SlaveParam):
        #        debug("{0}: {1} {2:.3f}".format(self.name, value.expr, value.value))
        #    else:
        #        debug("{0}: {1}".format(name, value.value))

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
        value = self.value
        unit_str = latex_unit(self.param.unit)
        return "${0}{1}$".format(latex(value), "\\;{0}".format(unit_str) \
                                 if unit_str else "")

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
        check_arg(init, scalars + (ScalarParam,), 1, State)

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
        check_arg(init, scalars + (ScalarParam,), 1, Parameter)

        # Call super class
        super(Parameter, self).__init__(name, init)

    init = ODEValueObject.value

class IndexedObject(ODEObject):
    """
    An object with a fixed index associated with it
    """
    def __init__(self, basename, indices, shape=None):
        """
        Create an IndexedExpression with an associated basename

        Arguments
        ---------
        basename : str
            The basename of the multi index Expression
        indices : tuple of ints
            The indices 
        shape : tuple (optional)
            A tuple with the shape of the indexed object
        """
        
        check_arg(basename, str)
        indices = tuplewrap(indices)
        check_arg(indices, tuple, itemtypes=int)

        # Get index format and index offset from global parameters
        index_format = parameters.code_generation.array.index_format
        index_offset = parameters.code_generation.array.index_offset
        flatten = parameters.code_generation.array.flatten
        
        # If trying to flatten indices, with a rank larger than 1 a shape needs
        # to be provided
        if len(indices)>1 and flatten and shape is None:
            error("A 'shape' need to be provided to generate flatten indices "\
                  "for index expressions with rank larger than 1.")

        # Create index format
        if index_format == "{}":
            index_format = "{{{0}}}"
        else:
            index_format = index_format[0]+"{0}"+index_format[1]

        # If flatten indices
        orig_indices = indices
        if flatten and len(indices)>1:
            indices = (sum(reduce(lambda i, j: i*j, shape[i+1:],1)*\
                           (index+index_offset) for i, index in \
                           enumerate(indices)),)
        else:
            indices = tuple(index+index_offset for index in indices)

        index_str = ",".join(str(index) for index in indices)
        name = basename + index_format.format(index_str)

        ODEObject.__init__(self, name)
        self._basename = basename
        self._indices = orig_indices
        self._sym = sp.Symbol(name)
        self._shape = shape

    @property
    def basename(self):
        return self._basename

    @property
    def indices(self):
        return self._indices

    @property
    def sym(self):
        return self._sym

    @property
    def shape(self):
        return self._shape

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
