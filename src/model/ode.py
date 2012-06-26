__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2012 " + __author__
__date__ = "2012-02-22 -- 2012-05-08"
__license__  = "GNU LGPL Version 3.0 or later"

__all__ = ["ODE", "gco"]

# System imports
import sympy as sym

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
        
        @type name : str
        @param name : Name of the ODE
        """
        global _current_ode

        check_arg(name, str, 0)

        # Set current Ode
        _current_ode = self

        # Initialize attributes
        self.name = name

        # Initialize all variables
        self.clear()

    def add_state(self, name, value):
        """
        Add a state to the ODE

        @type name : str
        @param name : The name of the state variable
        @type value : float, int
        @param value : Initial value of the state
        
        Example:
        ========

        >>> ode = ODE("MyOde")
        >>> ode.add_state("e", 1)
        """
        
        # Create the state
        state = State(self, name, value)
        
        # Register the state
        self._states.append(state)
        self._register_symbol(state)

        # Return the sympy version of the state
        return state.sym
        
    def add_field_state(self, name, value):
        """
        Add a field_state to the ODE

        @type name : str
        @param name : The name of the field state variable
        @type value : float, int, numpy array
        @param value : Initial value of the state
        
        Example:
        ========

        >>> ode = ODE("MyOde")
        >>> ode.add_field_state("v", 0.0)
        """
        
        # Create the field_state
        field_state = FieldState(self, name, value)
        
        # Register the field_state
        self._field_states.append(field_state)
        self._register_symbol(field_state)

        # Return the sympy version of the field state
        return field_state.sym
        
    def add_parameter(self, name, value):
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
        parameter = Parameter(self, name, value)
        
        # Register the parameter
        self._parameters.append(parameter)
        self._register_symbol(parameter)

        # Return the sympy version of the parameter
        return parameter.sym

    def add_variable(self, name, value):
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
        variable = Variable(self, name, value)
        
        # Register the variable
        self._variables.append(variable)
        self._register_symbol(variable)

        # Return the sympy version of the variable
        return variable.sym

    def get_symbol(self, name):
        """
        Return a registered symbol
        """
        
        return self._all_symbols.get(str(name))

    def diff(self, state, expr, dependent=t):
        """
        Register an expression for a state derivative
        """
        check_arg(state, sym.Symbol, 0)
        
        if not self.has_state(state):
            error("expected derivative of a declared state or field state")

        # Register the derivative
        self.get_symbol(str(state)).diff(expr, dependent)

    def iterstates(self):
        """
        Return an iterator over registered states
        """
        for state in self._states + self._field_states:
            yield state

    def has_state(self, state):
        """
        Return True if state is a registered state or field state
        """
        return str(state) in self._states or str(state) in self._field_states
        
    def has_field_state(self, state):
        """
        Return True if state is a registered field state
        """
        return str(state) in self._field_states
        
    def has_variable(self, param):
        """
        Return True if state is a registered variable
        """
        return str(param) in self._variables
        
    def has_parameter(self, state):
        """
        Return True if state is a registered parameter
        """
        return str(state) in self._parameters
        
    def is_complete(self):
        """
        Check that the ODE is complete
        """
        all_states = [state for state in self.iterstates()]

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
        # By default only t is a registered symbol
        return len(self._all_symbols) == 1

    def clear(self):
        """
        Clear any registered symbols
        """

        # FIXME: Make this a dict of lists
        self._states = []
        self._field_states = []
        self._parameters = []
        self._variables = []
        self._all_symbols = {}
        self._state_derivatives = {}

        # Add time as a variable
        self.add_variable("t", 0.0)

    def _register_symbol(self, symbol):
        """
        Register a symbol to the ODE
        """
        assert(isinstance(symbol, ODESymbol))
        
        # Make symbol available as an attribute
        setattr(self, str(symbol), symbol.sym)

        # Register the symbol
        self._all_symbols[str(symbol)] = symbol

    def _sort_collected_symbols(self):
        """
        Sort the collected symbols after alphabetic order
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
        check_arg(other, ODE)
        
        # Sort all collected symbols
        self._sort_collected_symbols()
        other._sort_collected_symbols()

        # Compare the list of symbols
        for what in ["_states", "_field_states", "_parameters", "_variables"]:
            if getattr(self, what) != getattr(other, what):
                return False

        # Check equal differentiation
        for state in self.iterstates():
            if len(state.diff_expr) != len(other.get_symbol(state).diff_expr):
                return False
            for dependent, expr in state.diff_expr.items():
                if dependent not in other.get_symbol(state).diff_expr:
                    return False
                if other.get_symbol(state).diff_expr[dependent] != \
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
    
