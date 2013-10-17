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

__all__ = ["Expression", "Intermediate", "DerivativeExpression"]

# ModelParameters imports
from modelparameters.sympytools import sp, iter_symbol_params_from_expr

# Local imports
from gotran.common import error, check_arg, scalars, Timer
from gotran.model.odeobjects import *
from gotran.model.odecomponents import *

class Expression(ValueODEObject):
    """
    class for all expressions such as intermediates and diff 
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
        
        if not any(isinstance(atom, (sp.Symbol, sp.Number)) \
                   for atom in expr.atoms()):
            error("expected the expression to contain at least one "\
                  "Symbol or Number.")

        # Call super class with expression as the "value"
        super(Expression, self).__init__(name, expr)
        
        self._expanded_expr = None

    @property
    def expr(self):
        """
        Return the stored expression
        """
        return self._param.expr

    @property
    def expanded_expr(self):
        """
        Return the stored expression
        """
        return self._expanded_expr

    @property
    def object_dependencies(self):
        """
        Return the object dependencies
        """
        return self._object_dependencies

class DerivativeExpression(Expression):
    """
    class for all derivative expressions
    """
    def __init__(self, derivatives, expr, ode, component=""):
        """
        Create an derivative or algebraic expression

        Arguments
        ---------
        derivatives : int, sympy.Basic
            A linear expression of StateDerivative symbols. Can also be an
            integer and then only 0, to add an algebraic expression
        expr : sympy.Basic
            The expression 
        ode : ODE
            The ODE which the derivative expression is declared within
        component : str (optional)
            The component will be determined automatically if the
            DerivativeExpression is an Algebraic expression
        """

        check_arg(derivatives, (sp.Basic, sp.Symbol, int), 0)

        error_str = "expected a linear combination of state derivatives "\
                    "as the derivative argument."

        def check_single_model_sym(sym):
            obj = ode.get_object(derivatives)
            if obj is None or not isinstance(obj, StateDerivative):
                error(error_str)
        
        def check_mul(mul):
            """
            Help function to check a mul operator in the derivative expression
            """
            derivative = None
            for arg in mul.args:
                if isinstance(arg, sp.Symbol):
                    obj = ode.get_object(arg)

                    if isinstance(obj, StateDerivative):

                        # Check that we have no registered Derivative 
                        if derivative is not None:
                            error(error_str)

                        # Save StateDerivative
                        derivative = obj
                        
                    elif isinstance(obj, Parameter):
                        # Parameters are fine
                        pass
                    else:
                        error(error_str)
                        
                elif arg.is_number:
                    # Numbers are fine
                    pass
                else:
                    error(error_str)

            # Check that we got one StateDerivative
            if derivative is None:
                error(error_str)

            # Return StateDerivative
            return derivative

        # Start checking derivatives

        stripped_derivatives = []

        # If an int we expect that to be Zero
        if isinstance(derivatives, int):
            if derivatives != 0:
                type_error("expected either an expression of derivatives or 0 "
                           "as the derivative arguments")
            name = "algebra"
  
        # If an expression of derivatives
        elif isinstance(derivatives, sp.Basic):

            # If single sp.Symbol we expect a DerivativeExpression
            if isinstance(derivatives, sp.Symbol):
                check_single_model_sym(derivatives)
                stripped_derivatives.append(ode.get_object(derivatives))

            # If derivatives is a linear combination of derivatives
            elif isinstance(derivatives, sp.Add):
                for arg in derivatives.args:
                    if isinstance(arg, sp.Symbol):
                        check_single_model_sym(arg)
                    elif isinstance(arg, sp.Mul):
                        stripped_derivatives.append(check_mul(arg))

            # If derivatives is a constantly weighted single derivative
            elif isinstance(derivatives, sp.Mul):
                derivative = check_mul(derivatives)
                warning("Got constant weighted single derivative. Divide "\
                        "whole expression with constant part.")
                constant_part = derivatives/derivative.sym
                expr = expr/constant_part
                derivatives = derivative.sym
                stripped_derivatives.append(derivative)
                
            else:
                type_error("expected a linear combination of "\
                           "derivatives as the derivative argument.")

            # Create name based on derivatives
            name = str(derivatives)

        # Store derivatives
        self._stripped_derivatives = stripped_derivatives
        self._derivatives = derivatives

        # Check that all derivative states belong to the same component
        if len(stripped_derivatives) == 1:
            component = stripped_derivatives[0].component
        elif len(stripped_derivatives) > 1:
            component = stripped_derivatives[0].component
            if not all(component == der.component \
                       for der in stripped_derivatives[1:]):
                error("Expected all derivative expressions to belong to "\
                      "the same component")
        else:
            # No derivative expression, use component passed to
            # constructor
            pass

        # Call super class with expression as the "init" value
        super(DerivativeExpression, self).__init__(name, expr, ode, component, slaved)

    @property
    def num_derivatives(self):
        """
        Return the number of derivatives
        """
        return len(self._stripped_derivatives)

    @property
    def derivatives(self):
        """
        Return the derivatives
        """
        return self._derivatives

    def stripped_derivatives(self):
        """
        Return a list of all derivatives
        """
        return self._stripped_derivatives

    @property
    def states(self):
        """
        Return the derivative states
        """
        return [der.state for der in self._stripped_derivatives]

    @property
    def is_algebraic(self):
        return not bool(self._stripped_derivatives)
    
class Intermediate(Expression):
    """
    class for all Intermediates 
    """
    def __init__(self, name, expr, ode, component="", slaved=False):
        """
        Create an Intermediate with an assosciated name

        Arguments
        ---------
        name : str
            The name of the Intermediate
        expr : sympy.Basic
            The expression
        ode : ODE
            The ODE which the Intermediate is declared within
        component : str (optional)
            A component for which the Intermediate should be associated with.
        slaved : bool
            If True the creation and differentiation is controlled by
            other entity, like a Markov model.
        """

        # Call super class with expression as the "init" value
        super(Intermediate, self).__init__(name, expr, ode, component, slaved)

