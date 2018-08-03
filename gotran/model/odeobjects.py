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
           "SingleODEObjects", "Time", "Dt", "IndexedObject", "cmp_to_key", "cmp"]

# System imports
import numpy as np
from collections import OrderedDict, defaultdict
import types

# ModelParameters imports
from modelparameters.sympytools import sp
from modelparameters.codegeneration import sympycode, latex
from modelparameters.parameters import *
from modelparameters.parameterdict import cmp_to_key

from gotran.common import error, check_arg, scalars, debug, DEBUG, \
     get_log_level, Timer, parameters
from functools import reduce, cmp_to_key


def cmp(a, b):
    return (a > b) - (a < b)

class ODEObject(object):
    """
    Base container class for all ODEObjects
    """
    __count = 0
    __dependent_counts = defaultdict(lambda : 0)
    def __init__(self, name, dependent=None):
        """
        Create ODEObject instance

        Arguments
        ---------
        name : str
            The name of the ODEObject
        dependent : ODEObject
            If given the count of this ODEObject will follow as a
            fractional count based on the count of the dependent object
        """

        
        check_arg(name, str, 0, ODEObject)
        check_arg(dependent, (type(None), ODEObject))
        self._name = self._check_name(name)

        # Unique identifyer
        if dependent is None:
            self._count = ODEObject.__count
            ODEObject.__count += 1
        else:
            ODEObject.__dependent_counts[dependent._count] += 1

            # FIXME: Do not hardcode the fractional increase
            self._count = dependent._count + \
                          ODEObject.__dependent_counts[dependent._count] * 0.00001
            
    def __hash__(self):
        return id(self)

    def __cmp__(self, other):
        if not isinstance(other, ODEObject):
            return -1
        return cmp(self._count, other._count)

    def __lt__(self, other):
        if not isinstance(other, ODEObject):
            return False
        return cmp(self._count, other._count) < 0

    def __gt__(self, other):
        if not isinstance(other, ODEObject):
            return False
        return cmp(self._count, other._count) > 0

    def __eq__(self, other):
        if not isinstance(other, ODEObject):
            return False
        return cmp(self._count, other._count) == 0

    def __le__(self, other):
        if not isinstance(other, ODEObject):
            return False
        return cmp(self._count, other._count) <= 0

    def __ge__(self, other):
        if not isinstance(other, ODEObject):
            return False
        return cmp(self._count, other._count) >= 0

    def __ne__(self, other):
        if not isinstance(other, ODEObject):
            return False
        return cmp(self._count, other._count) != 0

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

    def _recount(self, new_count=None, dependent=None):
        """
        Method called when an object need to get a new count
        """
        old_count = self._count

        if isinstance(new_count, (int, float)):
            check_arg(new_count, (int, float), ge=0, le=ODEObject.__count)
            self._count = new_count
        elif isinstance(dependent, ODEObject):
            ODEObject.__dependent_counts[dependent._count] += 1

            # FIXME: Do not hardcode the fractional increase
            self._count = dependent._count + \
                          ODEObject.__dependent_counts[dependent._count] * 0.00001
        else:
            self._count = ODEObject.__count
            ODEObject.__count += 1
            
        debug("Change count of {0} from {1} to {2}".format(\
            self.name, old_count, self._count))
    
class Comment(ODEObject):
    """
    A Comment. To keep track of user comments in an ODE
    """
    def __init__(self, comment, dependent=None):
        """
        Create a comment

        Arguments
        ---------
        comment : str
            The comment
        dependent : ODEObject
            If given the count of this comment will follow as a
            fractional count based on the count of the dependent object
        """

        # Call super class
        super(Comment, self).__init__(comment, dependent)

    def _repr_latex_(self):
        return "$\\mathbf{{{0}}}$".format(self.name.replace(" ", "\\;"))

class ODEValueObject(ODEObject):
    """
    A class for all ODE objects which has a value
    """
    def __init__(self, name, value, dependent=None):
        """
        Create ODEObject instance

        Arguments
        ---------
        name : str
            The name of the ODEObject
        value : scalar, ScalarParam, np.ndarray, sp. Basic
            The value of this ODEObject
        dependent : ODEObject
            If given the count of this ODEValueObject will follow as a
            fractional count based on the count of the dependent object
        """

        check_arg(name, str, 0, ODEValueObject)
        check_arg(value, scalars + (ScalarParam, sp.Basic), 1, ODEValueObject)

        # Init super class
        super(ODEValueObject, self).__init__(name, dependent)

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

    def update(self, value):
        self._param.update(value)
    
    @value.setter
    def value(self, value):
        self._param.setvalue(value)

    @property
    def sym(self):
        return self._param.sym

    @property
    def unit(self):
        return self._param.unit
    
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
        from modelparameters.codegeneration import latex_unit
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
        self.sym_0._assumptions["real"] = True
        self.sym_0._assumptions["imaginary"] = False
        self.sym_0._assumptions["commutative"] = True
        self.sym_0._assumptions["hermitian"] = True
        self.sym_0._assumptions["complex"] = True

        self._sym = self._param.sym(self.time.sym)
        self._sym._assumptions["real"] = True
        self._sym._assumptions["imaginary"] = False
        self._sym._assumptions["commutative"] = True
        self._sym._assumptions["hermitian"] = True
        self._sym._assumptions["complex"] = True

        # Flag to determine if State is solved or not
        self._is_solved = False

    init = ODEValueObject.value

    @property
    def sym(self):
        return self._sym

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

    
    def _get_binary_operands(self, x):
        
        # Check what type x is
        if type(x) in scalars:
            return (self.value, float(x))
        else:
            # Only allow for Parameter class
            # TODO: allow for other types of objects
            # in the modelparameter repo
            check_arg(x, Parameter)

            # TODO: Implement functionality for unit
            # conversion
            msg = ("Addition not implemented for "+
                   "parameters with different units")
            assert x.param.unit == self.param.unit, msg
        
            return (self.value,  x.value)

            
    def __add__(self, x):
        """
        Method wrapper for addition
        """
        x1,x2 = self._get_binary_operands(x)
        return x1 + x2
    
    def __radd__(self, x):
        """
        Method wrapper for addition
        """
        return self.__add__(x)
    
    def __mul__(self, x):
        """
        Method wrapper fro multiplication
        """
        x1,x2 = self._get_binary_operands(x)
        return x1 * x2

    def __rmul__(self, x):
        return self.__mul__(x)

    init = ODEValueObject.value

class IndexedObject(ODEObject):
    """
    An object with a fixed index associated with it
    """
    @staticmethod
    def default_parameters():
        """
        Return the default parameters for formating the array
        """
        return parameters.generation.code.array.copy()
    
    def __init__(self, basename, indices, shape=None, array_params=None, \
                 add_offset="", dependent=None):
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
        array_params : dict
            Parameters to create the array name for the indexed object
        add_offset : bool, str
            If True a fixed offset is added to the indices
        dependent : ODEObject
            If given the count of this IndexedObject will follow as a
            fractional count based on the count of the dependent object
        """

        array_params = array_params or {}
        
        check_arg(basename, str)
        indices = tuplewrap(indices)
        check_arg(indices, tuple, itemtypes=int)
        check_kwarg(array_params, "array_params", dict)

        # Get index format and index offset from global parameters
        self._array_params = self.default_parameters()
        self._array_params.update(array_params)
        
        index_format = self._array_params.index_format
        index_offset = self._array_params.index_offset
        flatten = self._array_params.flatten
        if add_offset and isinstance(add_offset, str):
            offset_str = add_offset
        else:
            offset_str = "{0}_offset".format(basename) if add_offset else ""
        self._offset_str = offset_str

        if offset_str:
            offset_str +="+"

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

        index_str = ",".join(offset_str+str(index) for index in indices)
        name = basename + index_format.format(index_str)

        ODEObject.__init__(self, name, dependent)
        self._basename = basename
        self._indices = orig_indices
        self._sym = sp.Symbol(name, real=True, imaginary=False, commutative=True,
                              hermitian=True, complex=True)
        self._shape = shape

    @property
    def offset_str(self):
        return self._offset_str

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
        self.sym_0 = sp.Symbol("{0}_0".format(name), real=True, imaginary=False,
                               commutative=True, hermitian=True, complex=True)

    def update_unit(self, unit):
        """
        Update unit. For example if time has unit
        seconds in stead of milliseconds
        """
      
        if unit in ["ms", "s"]:
            self._param = ScalarParam(0.0, unit=unit, name=self._name)
        else:
            info("Unknow time unit {}".format(unit))
        

class Dt(ODEValueObject):
    """
    Specialization for a time step class
    """
    def __init__(self, name, unit="ms"):
        super(Dt, self).__init__(name, ScalarParam(0.1, unit=unit))

    def update_unit(self, unit):
        """
        Update unit. For example if time has unit
        seconds in stead of milliseconds
        """
        if unit == "ms":
            self._param = ScalarParam(0.1, unit=unit, name=self._name)
        elif unit == "s":
            self._param = ScalarParam(100, unit=unit, name=self._name)
        else:
            info("Unknow time unit {}".format(unit))


# Tuple with single ODE Objects, for type checking
SingleODEObjects = (State, Parameter, Time)
