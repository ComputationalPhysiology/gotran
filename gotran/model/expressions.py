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

__all__ = [
    "Expression",
    "DerivativeExpression",
    "AlgebraicExpression",
    "StateExpression",
    "StateSolution",
    "RateExpression",
    "Intermediate",
    "StateDerivative",
    "Derivatives",
    "IndexedExpression",
    "StateIndexedExpression",
    "ParameterIndexedExpression",
    "recreate_expression",
]

from modelparameters.codegeneration import latex, sympycode
from modelparameters.logger import error

# ModelParameters imports
from modelparameters.parameters import SlaveParam
from modelparameters.sympytools import sp
from modelparameters.utils import check_arg, scalars

from .odeobjects import (
    IndexedObject,
    ODEValueObject,
    ParameterIndexedObject,
    State,
    StateIndexedObject,
    Time,
    cmp,
    cmp_to_key,
)


def recreate_expression(expr, *replace_dicts, **kwargs):
    """
    Recreate an Expression while applying replace dicts in given order
    """

    replace_type = kwargs.get("replace_type", "xreplace")
    if replace_type not in ["xreplace", "subs"]:
        error(
            "Valid alternatives for replace_type is: 'xreplace', "
            "'subs' got {0}".format(replace_type),
        )

    # First do the replacements
    sympyexpr = expr.expr
    for replace_dict in replace_dicts:
        if replace_type == "xreplace":
            sympyexpr = sympyexpr.xreplace(replace_dict)
        else:
            sympyexpr = sympyexpr.subs(replace_dict)

    # FIXME: Should we distinguish between the different
    # FIXME: intermediates?
    if isinstance(expr, Intermediate):
        new_expr = Intermediate(expr.name, sympyexpr)

    elif isinstance(expr, StateDerivative):
        new_expr = StateDerivative(expr.state, sympyexpr)

    elif isinstance(expr, AlgebraicExpression):
        new_expr = AlgebraicExpression(expr.state, sympyexpr)

    elif isinstance(expr, IndexedExpression):
        new_expr = IndexedExpression(
            expr.basename,
            expr.indices,
            sympyexpr,
            expr.shape,
            expr._array_params,
            expr._offset_str,
            enum=expr.enum,
        )
    elif isinstance(expr, StateIndexedExpression):
        new_expr = StateIndexedExpression(
            expr.basename,
            expr.indices,
            sympyexpr,
            expr.state,
            expr.shape,
            expr._array_params,
            expr._offset_str,
        )
    elif isinstance(expr, ParameterIndexedExpression):
        new_expr = ParameterIndexedExpression(
            expr.basename,
            expr.indices,
            sympyexpr,
            expr.state,
            expr.parameter,
            expr._array_params,
            expr._offset_str,
        )
    else:
        error("Should not reach here")

    # Inherit the count of the old expression
    new_expr._recount(expr._count)

    return new_expr


class Expression(ODEValueObject):
    """
    class for all expressions such as intermediates and derivatives
    """

    def __init__(self, name, expr, dependent=None):
        """
        Create an Expression with an associated name

        Arguments
        ---------
        name : str
            The name of the Expression
        expr : sympy.Basic
            The expression
        dependent : ODEObject
            If given the count of this Expression will follow as a
            fractional count based on the count of the dependent object
        """

        from modelparameters.sympytools import symbols_from_expr

        # Check arguments
        check_arg(expr, scalars + (sp.Basic,), 1, Expression)

        expr = sp.sympify(expr)

        # Deal with Subs in sympy expression
        for sub_expr in expr.atoms(sp.Subs):

            # deal with one Subs at a time
            subs = dict(
                (key, value) for key, value in zip(sub_expr.variables, sub_expr.point)
            )

            expr = expr.subs(sub_expr, sub_expr.expr.xreplace(subs))

        # Deal with im and re
        im_exprs = expr.atoms(sp.im)
        re_exprs = expr.atoms(sp.re)
        if im_exprs or re_exprs:
            replace_dict = {}
            for im_expr in im_exprs:
                replace_dict[im_expr] = sp.S.Zero
            for re_expr in re_exprs:
                replace_dict[re_expr] = re_expr.args[0]
            expr = expr.xreplace(replace_dict)

        if not symbols_from_expr(expr, include_numbers=True):
            error(
                "expected the expression to contain at least one " "Symbol or Number.",
            )

        # Call super class with expression as the "value"
        super(Expression, self).__init__(name, expr, dependent)

        # Collect dependent symbols
        dependent = tuple(
            sorted(
                symbols_from_expr(expr),
                key=cmp_to_key(lambda a, b: cmp(sympycode(a), sympycode(b))),
            ),
        )

        if dependent:
            self._sym = self._param.sym(*dependent)
            self._sym._assumptions["real"] = True
            self._sym._assumptions["commutative"] = True
            self._sym._assumptions["imaginary"] = False
            self._sym._assumptions["hermitian"] = True
            self._sym._assumptions["complex"] = True
        else:
            self._sym = self.param.sym

        self._dependent = set(dependent)

    @property
    def dependent(self):
        return self._dependent

    @property
    def expr(self):
        """
        Return the stored expression
        """
        return self._param.expr

    @property
    def sym(self):
        """"""
        return self._sym

    def replace_expr(self, *replace_dicts):
        """
        Replace registered expression using passed replace_dicts
        """
        expr = self.expr
        for replace_dict in replace_dicts:
            expr = expr.xreplace(replace_dict)
        self._param = SlaveParam(expr, name=self.name)

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
        return f"'{self.name}', {sympycode(self.expr)}"

    def _repr_latex_(self):
        """
        Return a pretty latex representation of the Expression object
        """

        return f"${self._repr_latex_name()} = {self._repr_latex_expr()}$"

    def _repr_latex_expr(self):
        return latex(self.expr)

    def _repr_latex_name(self):
        return f"{latex(self.name)}"


class Intermediate(Expression):
    """
    A class for all Intermediate classes
    """

    def __init__(self, name, expr, dependent=None):
        """
        Create an Intermediate with an associated name

        Arguments
        ---------
        name : str
            The name of the Expression
        expr : sympy.Basic
            The expression
        dependent : ODEObject
            If given the count of this Intermediate will follow as a
            fractional count based on the count of the dependent object
        """
        super(Intermediate, self).__init__(name, expr, dependent)


class StateSolution(Intermediate):
    """
    Sub class of Expression for state solution expressions
    """

    def __init__(self, state, expr, dependent=None):
        """
        Create a StateSolution

        Arguments
        ---------
        state : State
            The state that is being solved for
        expr : sympy.Basic
            The expression that should equal 0 and which solves the state
        dependent : ODEObject
            If given the count of this StateSolution will follow as a
            fractional count based on the count of the dependent object
        """

        check_arg(state, State, 0, StateSolution)
        super(StateSolution, self).__init__(sympycode(state.sym), expr)

        # Flag solved state
        state._is_solved = True
        self._state = state

    @property
    def state(self):
        return self._state

    def _args_str(self):
        """
        Return a formatted str of __init__ arguments
        """
        return f"'{repr(self.state)}', {sympycode(self.expr)}"


class DerivativeExpression(Intermediate):
    """
    A class for Intermediate derivative expressions
    """

    def __init__(self, der_expr, dep_var, expr, dependent=None):
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
        dependent : ODEObject
            If given the count of this DerivativeExpression will follow as a
            fractional count based on the count of the dependent object
        """
        check_arg(der_expr, Expression, 0, DerivativeExpression)
        check_arg(dep_var, (State, Expression, Time), 1, DerivativeExpression)

        # Check that the der_expr is dependent on var
        if dep_var.sym not in der_expr.sym.args:
            error(
                "Cannot create a DerivativeExpression as {0} is not "
                "dependent on {1}".format(der_expr, dep_var),
            )

        der_sym = sp.Derivative(der_expr.sym, dep_var.sym)
        self._der_expr = der_expr
        self._dep_var = dep_var

        super(DerivativeExpression, self).__init__(sympycode(der_sym), expr, dependent)
        self._sym = sp.Derivative(der_expr.sym, dep_var.sym)
        self._sym._assumptions["real"] = True
        self._sym._assumptions["commutative"] = True
        self._sym._assumptions["imaginary"] = False
        self._sym._assumptions["hermitian"] = True
        self._sym._assumptions["complex"] = True

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
        return f"{repr(self._der_expr)}, {repr(self._dep_var)}, {sympycode(self.expr)}"

    def _repr_latex_name(self):
        return "\\frac{{d{0}}}{{d{1}}}".format(
            latex(self._der_expr.name),
            latex(self._dep_var.name),
        )


class RateExpression(Intermediate):
    """
    A sub class of Expression holding single rates
    """

    def __init__(self, to_state, from_state, expr, dependent=None):

        check_arg(to_state, (State, StateSolution), 0, RateExpression)
        check_arg(from_state, (State, StateSolution), 1, RateExpression)

        super(RateExpression, self).__init__(
            f"rates_{to_state}_{from_state}",
            expr,
            dependent,
        )
        self._to_state = to_state
        self._from_state = from_state

    def _args_str(self):
        """
        Return a formatted str of __init__ arguments
        """
        return "{0}, {1}, {2}".format(
            repr(self._to_state),
            repr(self._from_state),
            sympycode(self.expr),
        )

    @property
    def states(self):
        """
        Return a tuple of the two states the rate expression describes the rate
        between
        """
        return self._to_state, self._from_state


class StateExpression(Expression):
    """
    An expression which determines a State.
    """

    def __init__(self, name, state, expr, dependent=None):
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
        dependent : ODEObject
            If given the count of this StateExpression will follow as a
            fractional count based on the count of the dependent object
        """

        check_arg(state, State, 0, StateExpression)

        super(StateExpression, self).__init__(name, expr, dependent)
        self._state = state

    @property
    def state(self):
        return self._state

    def _args_str(self):
        """
        Return a formatted str of __init__ arguments
        """
        return f"'{self.name}', {repr(self.state)}, {sympycode(self.expr)}"


class StateDerivative(StateExpression):
    """
    A class for all state derivatives
    """

    def __init__(self, state, expr, dependent=None):
        """
        Create a StateDerivative

        Arguments
        ---------
        state : State
            The state for which the StateDerivative should apply
        expr : sympy.Basic
            The expression which the differetiation should be equal
        dependent : ODEObject
            If given the count of this StateDerivative will follow as a
            fractional count based on the count of the dependent object
        """

        check_arg(state, State, 0, StateDerivative)
        sym = sp.Derivative(state.sym, state.time.sym)
        sym._assumptions["real"] = True
        sym._assumptions["imaginary"] = False
        sym._assumptions["commutative"] = True
        sym._assumptions["hermitian"] = True
        sym._assumptions["complex"] = True

        # Call base class constructor
        super(StateDerivative, self).__init__(sympycode(sym), state, expr, dependent)
        self._sym = sym

    @property
    def sym(self):
        return self._sym

    def _args_str(self):
        """
        Return a formatted str of __init__ arguments
        """
        return f"{repr(self._state)}, {sympycode(self.expr)}"

    def _repr_latex_name(self):
        return f"\\frac{{d{latex(self.state.name)}}}{{dt}}"


class AlgebraicExpression(StateExpression):
    """
    A class for algebraic expressions which relates a State with an
    expression which should equal to 0
    """

    def __init__(self, state, expr, dependent=None):
        """
        Create an AlgebraicExpression

        Arguments
        ---------
        state : State
            The State which the algebraic expression should determine
        expr : sympy.Basic
            The expression that should equal 0
        dependent : ODEObject
            If given the count of this StateDerivative will follow as a
            fractional count based on the count of the dependent object
        """
        check_arg(state, State, 0, AlgebraicExpression)

        super(AlgebraicExpression, self).__init__(
            f"alg_{state}_0",
            state,
            expr,
            dependent,
        )

        # Check that the expr is dependent on the state
        # FIXME: No need because Simone says so!
        # if state.sym not in self.sym:
        #    error("Cannot create an AlgebraicExpression as {0} is not "\
        #          "dependent on {1}".format(state, expr))

    def _args_str(self):
        """
        Return a formatted str of __init__ arguments
        """
        return f"{repr(self._state)}, {sympycode(self.expr)}"

    def _repr_latex_name(self):
        return "0"


class IndexedExpression(IndexedObject, Expression):
    """
    An expression which represents an expression with a fixed index
    associated with it
    """

    def __init__(
        self,
        basename,
        indices,
        expr,
        shape=None,
        array_params=None,
        add_offset="",
        dependent=None,
        enum=None,
    ):
        """
        Create an IndexedExpression with an associated basename used in code
        generation.

        Arguments
        ---------
        basename : str
            The basename of the multi index Expression
        indices : tuple of ints
            The indices
        expr : sympy.Basic
            The expression
        shape : tuple (optional)
            A tuple with the shape of the indexed expression
        array_params : dict
            Parameters to create the array name for the indexed object
        add_offset : bool, str
            If True a fixed offset is added to the indices
        dependent : ODEObject
            If given the count of this IndexedExpression will follow as a
            fractional count based on the count of the dependent object
        enum : str
            String that can be used for enumeration
        """
        IndexedObject.__init__(
            self,
            basename,
            indices,
            shape,
            array_params,
            add_offset,
            dependent,
            enum,
        )
        Expression.__init__(self, self.name, expr, dependent)


class StateIndexedExpression(StateIndexedObject, Expression):
    """
    An expression which represents an expression with a fixed state index
    associated with it
    """

    def __init__(
        self,
        basename,
        indices,
        expr,
        state,
        shape=None,
        array_params=None,
        add_offset="",
        dependent=None,
    ):
        """
        Create an IndexedExpression with an associated basename used in code
        generation.

        Arguments
        ---------
        basename : str
            The basename of the multi index Expression
        indices : tuple of ints
            The indices
        expr : sympy.Basic
            The expression
        state : State
            The state the expression index corresponds to. Used for enumeration.
        shape : tuple (optional)
            A tuple with the shape of the indexed expression
        array_params : dict
            Parameters to create the array name for the indexed object
        add_offset : bool, str
            If True a fixed offset is added to the indices
        dependent : ODEObject
            If given the count of this IndexedExpression will follow as a
            fractional count based on the count of the dependent object

        """

        StateIndexedObject.__init__(
            self,
            basename,
            indices,
            state,
            shape,
            array_params,
            add_offset,
            dependent,
        )
        Expression.__init__(self, self.name, expr, dependent)


class ParameterIndexedExpression(ParameterIndexedObject, Expression):
    """
    An expression which represents an expression with a fixed state index
    associated with it
    """

    def __init__(
        self,
        basename,
        indices,
        expr,
        parameter,
        shape=None,
        array_params=None,
        add_offset="",
        dependent=None,
    ):
        """
        Create an IndexedExpression with an associated basename used in code
        generation.

        Arguments
        ---------
        basename : str
            The basename of the multi index Expression
        indices : tuple of ints
            The indices
        expr : sympy.Basic
            The expression
        parameter : Parameter
            The parameter the expression index corresponds to. Used for enumeration.
        shape : tuple (optional)
            A tuple with the shape of the indexed expression
        array_params : dict
            Parameters to create the array name for the indexed object
        add_offset : bool, str
            If True a fixed offset is added to the indices
        dependent : ODEObject
            If given the count of this IndexedExpression will follow as a
            fractional count based on the count of the dependent object
        """

        ParameterIndexedObject.__init__(
            self,
            basename,
            indices,
            parameter,
            shape,
            array_params,
            add_offset,
            dependent,
        )
        Expression.__init__(self, self.name, expr, dependent)


# Tuple with Derivative types, for type checking
Derivatives = (StateDerivative, DerivativeExpression)
