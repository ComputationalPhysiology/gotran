__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2012 " + __author__
__date__ = "2012-02-22 -- 2012-05-08"
__license__  = "GNU LGPL Version 3.0 or later"

__all__ = ["t", "ODESymbol", "Parameter", "State", "FieldState", "Variable"]#
#           "DiscreteVariable", "ContinuousVariable", ]

# System imports
import sympy
import numpy as np

# Gotran imports
from gotran2.common import error, check_arg

# Default time symbol
t = sympy.Symbol("t")

class ODESymbol(object):
    """
    Base container class for all symbols
    """
    def __init__(self, ode, name, value=None):
        """
        Create ODESymbol instance

        @type ode : ODE
        @param ode : The ode the symbol belongs to
        @type name : str
        @param name : The name of the ODESymbol
        @type value : float, int, None
        @param value : An optional numeric value of the symbol
        """
        from gotran2.model.ode import ODE
        check_arg(ode, ODE, 0)
        check_arg(name, str, 1)
        self.name = name
        self.ode = ode
        self.value = value
        self.sym = sympy.Symbol(name)

        if ode.get_symbol(name) is not None:
           error("Name '{}' is already registered in '{}'".format(ode.get_symbol(name), ode))

    def __eq__(self, other):
        """
        x.__eq__(y) <==> x==y
        """
        check_arg(other, (str, ODESymbol, sympy.Symbol))
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
        return "{}({})".format(self.__class__.__name__, self._args_str())

    def _args_str(self):
        """
        Return a formated str of __init__ arguments
        """
        return "'{}, '{}'{}'".format(repr(self.ode), self.name, "" \
                                   if self.value is None else \
                                   ", value={:f}".format(self.value))

class State(ODESymbol):
    """
    Container class for a State variable
    """
    def __init__(self, ode, name, value):
        """
        Create a state variable with an assosciated initial value

        @type ode : ODE
        @param ode : The ode the symbol belongs to
        @type name : str
        @param name : The name of the state variable
        @type value : float, int
        @param value : Initial value of the state
        """
        
        # Call super class
        check_arg(value, (float, int), 1)
        super(State, self).__init__(ode, name, value)

        # Add an attribute to register dependencies
        self.diff_expr = {}
        self.dependencies = {}
        self.linear_dependencies = {}

    def diff(self, expr, dependent=t):
        """
        Register a derivative of the state
        """
        check_arg(expr, (sympy.Basic, float, int), 0)
        check_arg(dependent, sympy.Symbol, 1)

        if str(dependent) in self.diff_expr:
            error("derivative of '{0}' is already registered.".format(self))

        self.dependencies[str(dependent)] = []
        self.linear_dependencies[str(dependent)] = []
        
        for atom in expr.atoms():
            if not isinstance(atom, (sympy.Atom, int, float)):
                error("a derivative must be sympy expressions or scalars")
            if isinstance(atom, sympy.Symbol):
                sym = self.ode.get_symbol(atom)

                if sym is None:
                    error("ODESymbol '{}' is not registered in the ""\
                    '{}' ODE".format(sym, self.ode))

                # Check dependencies on other states
                if self.ode.has_state(sym):
                    self.dependencies[str(dependent)].append(sym)
                    if atom not in expr.diff(atom).atoms():
                        self.linear_dependencies[str(dependent)].append(sym)

        # Store expression
        self.diff_expr[str(dependent)] = expr

    def has_diff(self, dependent=t):
        """
        Return True if differential is registered for dependent
        """
        return str(dependent) in self.diff_expr
        

class FieldState(State):
    """
    Container class for a FieldState variable
    """
    def __init__(self, ode, name, value):
        """
        Create a field state variable with an assosciated initial value

        @type ode : ODE
        @param ode : The ode the symbol belongs to
        @type name : str
        @param name : The name of the field state variable
        @type value : float, int, numpy array
        @param value : Initial value of the state
        """
        
        # Call super class
        check_arg(value, (float, int, np.ndarray), 1)
        super(FieldState, self).__init__(ode, name, value)

# FIXME: How to integrate a scalar parameter with OptionParameters, aso
class Parameter(ODESymbol):
    """
    Container class for a Parameter
    """
    def __init__(self, ode, name, value, ):
        """
        Create a Parameter with an assosciated initial value

        @type ode : ODE
        @param ode : The ode the symbol belongs to
        @type name : str
        @param name : The name of the parameter
        @type value : float, int
        @param value : Initial value of the parameter
        """
        
        # Call super class
        check_arg(value, (float, int), 1)
        super(Parameter, self).__init__(ode, name, value)

class Variable(ODESymbol):
    """
    Container class for a Variable
    """
    def __init__(self, ode, name, value):
        """
        Create a variable with an assosciated initial value

        @type ode : ODE
        @param ode : The ode the symbol belongs to
        @type name : str
        @param name : The name of the state variable
        @type value : float, int
        @param value : Initial value of the state
        """
        
        # Call super class
        check_arg(value, (float, int), 1)
        super(Variable, self).__init__(ode, name, value)


