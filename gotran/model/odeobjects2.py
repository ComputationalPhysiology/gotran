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
           "SingleODEObjects", "Time", "Dt", "Expression", \
           "DerivativeExpression", "AlgebraicExpression", \
           "StateSolution", "RateDerivative", "RateExpression"]

# System imports
import numpy as np
from collections import OrderedDict
from sympy.core.function import AppliedUndef

# ModelParameters imports
from modelparameters.sympytools import sp, symbols_from_expr, \
     iter_symbol_params_from_expr
from modelparameters.codegeneration import sympycode
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
        return "'{0}', {1}".format(self.name, self._param.repr(\
            include_name=False))

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
        Return a formated str of __init__ arguments
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


class Expression(ODEValueObject):
    """
    class for all expressions such as intermediates and derivatives
    """
    def __init__(self, name, expr):
        """
        Create an Exression with an assosciated name

        Arguments
        ---------
        name : str
            The name of the Expression
        expr : sympy.Basic
            The expression
        """

        # Check arguments
        from gotran.model.ode import ODE
        check_arg(expr, scalars + (sp.Basic,), 1, Expression)

        expr = sp.sympify(expr)

        # Deal with Subs in sympy expression
        for sub_expr in expr.atoms(sp.Subs):

            # deal with one Subs at a time
            subs = dict((key,value) for key, value in \
                    zip(sub_expr.variables, sub_expr.point))

            expr =  expr.subs(sub_expr, sub_expr.expr.xreplace(subs))

        if not symbols_from_expr(expr, include_numbers=True):
            error("expected the expression to contain at least one "\
                  "Symbol or Number.")

        # Collect dependent symbols
        self.dependent = tuple(symbols_from_expr(expr))

        self._is_state_expression = False

        # Call super class with expression as the "value"
        super(Expression, self).__init__(name, expr)

    @property
    def expr(self):
        """
        Return the stored expression
        """
        return self._param.expr

    @property
    def sym(self):
        """
        """
        if self.dependent:
            return self._param.sym(*self.dependent)
        else:
            return self._param.sym

    @property
    def is_state_expression(self):
        """
        True of expression is a state expression
        """
        return self._is_state_expression

    def _args_str(self):
        """
        Return a formated str of __init__ arguments
        """
        return "'{0}', {1}".format(self.name, repr(self.expr))

class DerivativeExpression(Expression):
    """
    Sub class for all derivative expressions
    """
    def __init__(self, der_expr, dep_var, expr):
        """
        Create a DerivativeExpression

        Arguments
        ---------
        der_expr : Expression, State
            The Expression or State which is differentiated
        dep_var : State, Time, Expression
            The dependent variable
        expr : sympy.Basic
            The expression which the differetiation should be equal
        """
        check_arg(der_expr, (State, Expression), 0, DerivativeExpression)
        check_arg(dep_var, (State, Expression, Time), 1, DerivativeExpression)

        # Check that the der_expr is dependent on var
        if dep_var.sym not in der_expr.sym:
            error("Cannot create a DerivativeExpression as {0} is not "\
                  "dependent on {1}".format(der_expr, dep_var))

        self._sym = sp.Derivative(der_expr.sym, dep_var.sym)
        self._der_expr = der_expr
        self._dep_var = dep_var

        super(DerivativeExpression, self).__init__(sympycode(self._sym), expr)

        if isinstance(der_expr, State):
            self._is_state_expression = True

    @property
    def sym(self):
        return self._sym

    @property
    def der_expr(self):
        return self._der_expr

    @property
    def dep_var(self):
        return self._dep_var

    def _args_str(self):
        """
        Return a formated str of __init__ arguments
        """
        return "{0}, {1}, {2}".format(repr(self._der_expr), repr(self._dep_var),\
                                      repr(self.expr))

class RateExpression(Expression):
    """
    A sub class of Expression holding single rates
    """
    def __init__(self, from_state, to_state, expr):

        check_arg(from_state, (State, StateSolution), 0, RateExpression)
        check_arg(to_state, (State, StateSolution), 1, RateExpression)

        super(RateExpression, self).__init__("{0}_{1}".format(\
            from_state, to_state), expr)
        self._from_state = from_state
        self._to_state = to_state

class RateDerivative(DerivativeExpression):
    """
    A state derivative expression for rate expressions
    """
    def __init__(self, state):
        """
        Create a RateDerivative

        Arguments
        ---------
        state : State
            The State the rate derivative determines
        """
        check_arg(state, State, 0, RateDerivative)

        # Initate the base class with a dummy expression
        super(RateDerivative, self).__init__(state, state.time, sp.sympify(0))

        # Variable which is used to build up the deriviative expression
        self._expr = sp.sympify(0)
        
    def add_rate(self, rate):
        """
        Register a rate expression to the Derivative
        """
        self._expr += rate
        
    @property
    def expr(self):
        """
        Return the stored expression
        """
        return self._expr

class AlgebraicExpression(Expression):
    """
    Sub class for all algebraic expressions which relates a State with
    an expression which should equal to 0
    """
    def __init__(self, state, expr):
        """
        Create an AlgebraicExpression

        Arguments
        ---------
        state : State
            The State which the algebraic expression should determine
        expr : sympy.Basic
            The expression that should equal 0
        """
        check_arg(state, State, 0, AlgebraicExpression)

        super(AlgebraicExpression, self).__init__("alg_{0}_0".format(state), expr)

        # Check that the expr is dependent on the state
        if state.sym not in self.sym:
            error("Cannot create an AlgebraicExpression as {0} is not "\
                  "dependent on {1}".format(state, expr))

        self._is_state_expression = True
        self._state = state

    @property
    def state(self):
        return self._state

    @property
    def _args_str(self):
        """
        Return a formated str of __init__ arguments
        """
        return "{0}, {1}".format(repr(self._state), repr(self.expr))

class StateSolution(Expression):
    """
    Sub class of Expression for state solution expressions
    """
    def __init__(self, state, expr):
        """
        Create a StateSolution

        Arguments
        ---------
        state : State
            The state that is being solved for
        expr : sympy.Basic
            The expression that should equal 0 and which solves the state
        """
        check_arg(state, State, 0, StateSolution)
        check_arg(expr, sp.Basic, 1, StateSolution)

        if state.is_field:
            error("Cannot registered a solved state that is a field_state")

        super(StateSolution, self).__init__(state.name, expr)

        # Flag solved state
        state._is_solved = True
        
        self._state = state

    @property
    def state(self):
        return self._state

    def _args_str(self):
        """
        Return a formated str of __init__ arguments
        """
        return "'{0}', {1}".format(repr(self.state), repr(self.expr))

# Tuple with single ODE Objects, for type checking
SingleODEObjects = (State, Parameter, Time)
