__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2012 " + __author__
__date__ = "2012-02-22 -- 2012-08-24"
__license__  = "GNU LGPL Version 3.0 or later"

__all__ = ["ODE", "gco"]

# System imports
import sympy as sym

# ModelParameter imports
from modelparameters.sympytools import ModelSymbol

# Gotran imports
from gotran2.common import error, check_arg
from gotran2.model.symbols import *

# Holder for current ODE
global _current_ode
_current_ode = None

class ODE(object):
    """
    Basic class for storying information of an ODE
    """
        
    def __init__(self, name):
        """
        Initialize an ODE
        
        Arguments
        ---------
        name : str
            The name of the ODE
        """
        global _current_ode

        check_arg(name, str, 0)

        # Set current Ode
        _current_ode = self

        # Initialize attributes
        self.name = name

        # Initialize all variables
        self.clear()

    def add_state(self, name, init):
        """
        Add a state to the ODE

        Arguments
        ---------
        name : str
            The name of the state variable
        init : scalar, ScalarParam
            The initial value of the state
        
        Example:
        ========

        >>> ode = ODE("MyOde")
        >>> ode.add_state("e", 1)
        """
        
        # Create the state
        state = State(self, name, init)
        
        # Register the state
        self._states.append(state)
        self._register_object(state)

        # Return the sympy version of the state
        return state.sym
        
    def add_parameter(self, name, init):
        """
        Add a parameter to the ODE

        @type name : str
        @param name : The name of the parameter
        @type value : float, int
        @param value : Initial value of the parameter
        
        Example:
        ========

        >>> ode = ODE("MyOde")
        >>> ode.add_parameter("c0", 5.0)
        """
        
        # Create the parameter
        parameter = Parameter(self, name, init)
        
        # Register the parameter
        self._parameters.append(parameter)
        self._register_object(parameter)

        # Return the sympy version of the parameter
        return parameter.sym

    def add_variable(self, name, init):
        """
        Add a variable to the ODE

        @type name : str
        @param name : The name of the variable
        @type value : float, int
        @param value : Initial value of the variable
        
        Example:
        ========

        >>> ode = ODE("MyOde")
        >>> ode.add_variable("c0", 5.0)
        """
        
        # Create the variable
        variable = Variable(self, name, init)
        
        # Register the variable
        self._variables.append(variable)
        self._register_object(variable)

        # Return the sympy version of the variable
        return variable.sym

    def get_object(self, name):
        """
        Return a registered object
        """

        check_arg(name, (str, ModelSymbol))
        if isinstance(name, ModelSymbol):
            name = name.name
        
        return self._all_objects.get(name)

    def diff(self, state, expr):
        """
        Register an expression for a state derivative
        """
        check_arg(state, ModelSymbol, 0)
        
        if not self.has_state(state):
            error("expected derivative of a declared state or field state")

        # Register the derivative
        self.get_object(state.name).diff(expr)

    def iter_states(self):
        """
        Return an iterator over registered states
        """
        for state in self._all_objects.values():
            if isinstance(state, State):
                yield state

    def iter_field_states(self):
        """
        Return an iterator over registered field states
        """
        for state in self._all_objects.values():
            if isinstance(state, State) and state.is_field:
                yield state

    def iter_variables(self):
        """
        Return an iterator over registered variables
        """
        for variable in self._all_objects.values():
            if isinstance(variable, Variable):
                yield variable

    def iter_parameters(self):
        """
        Return an iterator over registered parameters
        """
        for parameter in self._all_objects.values():
            if isinstance(parameter, Parameter):
                yield parameter

    def has_state(self, state):
        """
        Return True if state is a registered state or field state
        """
        check_arg(state, (str, ModelSymbol, ODEObject))
        if isinstance(state, (str, ModelSymbol)):
            state = self.get_object(state)
        
        if not isinstance(state, State):
            return False
        
        return state.ode == self
        
    def has_field_state(self, state):
        """
        Return True if state is a registered field state
        """
        check_arg(state, (str, ModelSymbol, ODEObject))
        if isinstance(state, (str, ModelSymbol)):
            state = self.get_object(state)
        
        if not isinstance(state, State):
            return False
        
        return state.is_field and state.ode == self
        
    def has_variable(self, variable):
        """
        Return True if variable is a registered variable
        """
        check_arg(variable, (str, ModelSymbol, ODEObject))
        if isinstance(variable, (str, ModelSymbol)):
            variable = self.get_object(variable)
        
        if not isinstance(Variable):
            return False
        
        return variable.ode == self
        
    def has_parameter(self, param):
        """
        Return True if state is a registered parameter
        """
        check_arg(param, (str, ModelSymbol, ODEObject))
        if isinstance(param, (str, ModelSymbol)):
            param = self.get_object(param)
        
        if not isinstance(Parameter):
            return False
        
        return param.ode == self

    @property
    def num_states(self):
        return len([s for s in self.iter_states()])
        
    @property
    def num_field_states(self):
        return len([s for s in self.iter_field_states()])
        
    @property
    def num_parameters(self):
        return len([s for s in self.iter_parameters()])
        
    @property
    def num_variables(self):
        return len([s for s in self.iter_variables()])
        
    def is_complete(self):
        """
        Check that the ODE is complete
        """
        all_states = [state for state in self.iter_states()]

        if not all_states:
            return False
        
        # FIXME: More thorough test?
        for state in all_states:
            if not state.has_diff():
                return False
        
        return True

    def is_empty(self):
        """
        Returns True if the ODE is empty
        """
        # By default only t is a registered object
        return len(self._all_objects) == 1

    def clear(self):
        """
        Clear any registered objects
        """

        # FIXME: Make this a dict of lists
        self._states = []
        self._field_states = []
        self._parameters = []
        self._variables = []
        self._all_objects = {}
        self._state_derivatives = {}

        # Add time as a variable
        self.add_variable("t", 0.0)
        self.add_variable("dt", 0.1)

    def _register_object(self, obj):
        """
        Register an object to the ODE
        """
        assert(isinstance(obj, ODEObject))
        
        # Make object available as an attribute
        setattr(self, obj.name, obj.sym)

        # Register the object
        self._all_objects[obj.name] = obj

    def _sort_collected_objects(self):
        """
        Sort the collected object after alphabetic order
        """ 
        cmp_func = lambda a, b: cmp(a.name, b.name)
        self._states.sort(cmp_func)
        self._field_states.sort(cmp_func)
        self._parameters.sort(cmp_func)
        self._variables.sort(cmp_func)

    def __eq__(self, other):
        """
        x.__eq__(y) <==> x==y
        """
        if not isinstance(other, ODE):
            return False

        if id(self) == id(other):
            return True
        
        # Sort all collected objects
        self._sort_collected_objects()
        other._sort_collected_objects()

        # Compare the list of objects
        for what in ["_states", "_field_states", "_parameters", "_variables"]:
            if getattr(self, what) != getattr(other, what):
                return False

        # Check equal differentiation
        # FIXME: Remove dependent
        for state in self.iter_states():
            if len(state.diff_expr) != len(other.get_object(state).diff_expr):
                return False
            for dependent, expr in state.diff_expr.items():
                if dependent not in other.get_object(state).diff_expr:
                    return False
                if other.get_object(state).diff_expr[dependent] != \
                   state.diff_expr[dependent]:
                    return False
        
        return True

    def __str__(self):
        """
        x.__str__() <==> str(x)
        """
        return self.name
        
    def __repr__(self):
        """
        x.__repr__() <==> repr(x)
        """
        return "{}('{}')".format(self.__class__.__name__, self.name)

# Construct a default Ode
_current_ode = ODE("Default")
        
def gco():
    """
    Return the current Ode
    """
    assert(isinstance(_current_ode, ODE))
    return _current_ode
    
