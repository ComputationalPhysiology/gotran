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

__all__ = ["Expression", "DerivativeExpression", "AlgebraicExpression", \
           "StateExpression", "StateSolution", "RateExpression", \
           "StateDerivative", "Derivatives"]

# ModelParameters imports
from modelparameters.sympytools import sp, symbols_from_expr
from modelparameters.codegeneration import sympycode

# Local imports
from gotran.common import error, check_arg, scalars, debug, DEBUG, \
     get_log_level, Timer
from gotran.model.odeobjects2 import *

class Expression(ODEValueObject):
    """
    class for all expressions such as intermediates and derivatives
    """
    def __init__(self, name, expr):
        """
        Create an Expression with an associated name

        Arguments
        ---------
        name : str
            The name of the Expression
        expr : sympy.Basic
            The expression
        """

        # Check arguments
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
        Return a formatted str of __init__ arguments
        """
        return "'{0}', {1}".format(self.name, repr(self.expr))

class StateExpression(Expression):
    """
    An expression which determines a State.
    """
    def __init__(self, name, state, expr):
        """
        Create an StateExpression with an assosiated name

        Arguments
        ---------
        name : str
            The name of the state expression. A symbol based on the name will
            be created and accessible to build other expressions
        state : State
            The state which the expression should determine
        expr : sympy.Basic
            The mathematical expression
        """

        check_arg(state, State, 0, StateExpression)

        super(StateExpression, self).__init__(name, expr)
        self._state = state
        
    @property
    def state(self):
        return self._state

    def _args_str(self):
        """
        Return a formatted str of __init__ arguments
        """
        return "'{0}', {1}, {2}".format(self.name, repr(state), sympycode(self.expr))

class StateDerivative(StateExpression):
    """
    A class for all state derivatives
    """
    def __init__(self, state, expr):
        """
        Create a StateDerivative

        Arguments
        ---------
        state : State
            The state for which the StateDerivative should apply
        expr : sympy.Basic
            The expression which the differetiation should be equal
        """
        
        check_arg(state, State, 0, StateDerivative)
        sym = sp.Derivative(state.sym, state.time.sym)

        # Call base class constructor
        super(StateDerivative, self).__init__(sympycode(sym), state, expr)
        self._sym = sym

    @property
    def sym(self):
        return self._sym

    def _args_str(self):
        """
        Return a formatted str of __init__ arguments
        """
        return "{0}, {1}".format(repr(self._state), sympycode(self.expr))

class AlgebraicExpression(StateExpression):
    """
    A class for algebraic expressions which relates a State with an
    expression which should equal to 0
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

        super(AlgebraicExpression, self).__init__("alg_{0}_0".format(\
            state), state, expr)

        # Check that the expr is dependent on the state
        if state.sym not in self.sym:
            error("Cannot create an AlgebraicExpression as {0} is not "\
                  "dependent on {1}".format(state, expr))

    def _args_str(self):
        """
        Return a formatted str of __init__ arguments
        """
        return "{0}, {1}".format(repr(self._state), repr(self.expr))

class StateSolution(StateExpression):
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
        super(StateSolution, self).__init__(sympycode(state.sym), state, expr)

        if state.is_field:
            error("Cannot registered a solved state that is a field_state")

        # Flag solved state
        state._is_solved = True
        
    def _args_str(self):
        """
        Return a formatted str of __init__ arguments
        """
        return "'{0}', {1}".format(repr(self.state), sympycode(self.expr))

class DerivativeExpression(Expression):
    """
    A class for Intermediate derivative expressions
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
        check_arg(der_expr, Expression, 0, DerivativeExpression)
        check_arg(dep_var, (State, Expression, Time), 1, DerivativeExpression)

        # Check that the der_expr is dependent on var
        if dep_var.sym not in der_expr.sym:
            error("Cannot create a DerivativeExpression as {0} is not "\
                  "dependent on {1}".format(der_expr, dep_var))

        self._sym = sp.Derivative(der_expr.sym, dep_var.sym)
        self._der_expr = der_expr
        self._dep_var = dep_var

        super(DerivativeExpression, self).__init__(sympycode(self._sym), expr)

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
        Return a formatted str of __init__ arguments
        """
        return "{0}, {1}, {2}".format(repr(self._der_expr), repr(self._dep_var),\
                                      self.expr)

class RateExpression(Expression):
    """
    A sub class of Expression holding single rates
    """
    def __init__(self, to_state, from_state, expr):

        check_arg(from_state, (State, StateSolution), 0, RateExpression)
        check_arg(to_state, (State, StateSolution), 1, RateExpression)

        super(RateExpression, self).__init__("rate_{0}_{1}".format(\
            to_state, from_state), expr)
        self._to_state = to_state
        self._from_state = from_state

# Tuple with Derivative types, for type checking
Derivatives = (StateDerivative, DerivativeExpression)
