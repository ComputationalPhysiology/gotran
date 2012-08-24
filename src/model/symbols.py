__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2012 " + __author__
__date__ = "2012-02-22 -- 2012-08-24"
__license__  = "GNU LGPL Version 3.0 or later"

__all__ = ["ODEObject", "Parameter", "State", "Variable"]#
#           "DiscreteVariable", "ContinuousVariable", ]

# System imports
import numpy as np

# ModelParameters imports
from modelparameters.sympytools import sp, ModelSymbol
from modelparameters.parameters import *

from gotran2.common import error, check_arg, scalars

class ODEObject(object):
    """
    Base container class for all symbols
    """
    def __init__(self, ode, name, init):
        """
        Create ODEObject instance

        ode : ODE
            The ODE the ODEObject belongs to
        name : str
            The name of the ODEObject
        init : scalar, ScalarParam
            The initial value of this ODEObject
        """

        from gotran2.model.ode import ODE
        check_arg(ode, ODE, 0, ODEObject)
        check_arg(name, str, 1, ODEObject)
        check_arg(init, scalars + (ScalarParam, list, np.ndarray), 2, ODEObject)
        
        self.ode = ode

        if isinstance(init, ScalarParam):

            # If Param already has a symbol
            if init.sym != dummy_sym:

                # Re create one without a name
                init = eval(repr(init).split(", name")[0]+")")

        elif isinstance(init, scalars):
            init = ScalarParam(init)
        else:
            init = ArrayParam(init)
            
        # Create a symname based on the name of the ODE
        init.name = name, "{0}.{1}".format(ode.name, name)

        # Store the Param
        self._param = init 

        # Store field
        # FIXME: Is this nesesary
        self._field = isinstance(init, ArrayParam)

        # Check that there are no object with this name
        if ode.get_object(name) is not None:
           error("Name '{0}' is already registered in '{1}'".format(\
               ode.get_object(name), ode))

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
        return "'{0}, '{1}', {2}{3}{4}'".format(\
            repr(self.ode), self.name, self.init, self._param._check_arg(),
            ", field=True" if self._field else "")

class State(ODEObject):
    """
    Container class for a State variable
    """
    def __init__(self, ode, name, init):
        """
        Create a state variable with an assosciated initial value

        ode : ODE
            The ODE the State belongs to
        name : str
            The name of the State
        init : scalar, ScalarParam
            The initial value of this state
        """
        
        # Call super class
        super(State, self).__init__(ode, name, init)

        # Add an attribute to register dependencies
        self.diff_expr = None
        self.dependencies = []
        self.linear_dependencies = []

        # Add previous value symbol
        self.sym_0 = ModelSymbol("{0}_0".format(name), \
                                 "{0}.{1}_0".format(ode.name, name))

    def diff(self, expr):
        """
        Register a derivative of the state
        """
        check_arg(expr, (sp.Basic, scalars), 0)

        for atom in expr.atoms():
            if not isinstance(atom, (ModelSymbol, sp.Number, int, float)):
                type_error("a derivative must be an expressions of "\
                           "ModelSymbol or scalars")
                
            if isinstance(atom, ModelSymbol):
                sym = self.ode.get_object(atom)

                if sym is None:
                    error("ODEObject '{0}' is not registered in the ""\
                    '{1}' ODE".format(sym, self.ode))

                # Check dependencies on other states
                if self.ode.has_state(sym):
                    self.dependencies.append(sym)
                    if atom not in expr.diff(atom).atoms():
                        self.linear_dependencies.append(sym)

        # Store expression
        self.diff_expr = expr

class Parameter(ODEObject):
    """
    Container class for a Parameter
    """
    def __init__(self, ode, name, init):
        """
        Create a Parameter with an assosciated initial value

        ode : ODE
            The ODE the State belongs to
        name : str
            The name of the State
        init : scalar, ScalarParam
            The initial value of this parameter
        """
        
        # Call super class
        super(Parameter, self).__init__(ode, name, init)

class Variable(ODEObject):
    """
    Container class for a Variable
    """
    def __init__(self, ode, name, init):
        """
        Create a variable with an assosciated initial value

        ode : ODE
            The ODE the Variable belongs to
        name : str
            The name of the variable
        init : scalar
            The initial value of this variable
        """
        
        # Call super class
        super(Variable, self).__init__(ode, name, init)

        # Add previous value symbol
        self.sym_0 = ModelSymbol("{0}__0".format(name), \
                                 "{0}.{1}__0".format(ode.name, name))


