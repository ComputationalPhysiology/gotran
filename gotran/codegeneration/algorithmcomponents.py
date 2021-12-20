# Copyright (C) 2013 Johan Hake
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
    "JacobianComponent",
    "JacobianActionComponent",
    "FactorizedJacobianComponent",
    "ForwardBackwardSubstitutionComponent",
    "LinearizedDerivativeComponent",
    "CommonSubExpressionODE",
    "componentwise_derivative",
    "linearized_derivatives",
    "jacobian_expressions",
    "jacobian_action_expressions",
    "factorized_jacobian_expressions",
    "forward_backward_subst_expressions",
    "diagonal_jacobian_expressions",
    "rhs_expressions",
    "diagonal_jacobian_action_expressions",
    "monitored_expressions",
]

# System imports
import sys

from modelparameters.codegeneration import sympycode
from modelparameters.logger import error, info

# ModelParameters imports
from modelparameters.sympytools import sp
from modelparameters.utils import Timer, check_arg, check_kwarg, listwrap
from modelparameters.sympy import cse

from gotran.model.expressions import (
    AlgebraicExpression,
    Expression,
    IndexedExpression,
    StateDerivative,
)

from ..model.ode import ODE
from ..model.odeobjects import Comment
from ..model.utils import ode_primitives
from .codecomponent import CodeComponent


def rhs_expressions(ode, function_name="rhs", result_name="dy", params=None):
    """
    Return a code component with body expressions for the right hand side

    Arguments
    ---------
    ode : gotran.ODE
        The finalized ODE
    function_name : str
        The name of the function which should be generated
    result_name : str
        The name of the variable storing the rhs result
    params : dict
        Parameters determining how the code should be generated
    """

    check_arg(ode, ODE)
    if not ode.is_finalized:
        error(
            "Cannot compute right hand side expressions if the ODE is " "not finalized",
        )

    descr = f"Compute the right hand side of the {ode} ODE"

    return CodeComponent(
        "RHSComponent",
        ode,
        function_name,
        descr,
        params=params,
        **{result_name: ode.state_expressions},
    )


def monitored_expressions(
    ode,
    monitored,
    function_name="monitored_expressions",
    result_name="monitored",
    params=None,
):
    """
    Return a code component with body expressions to calculate monitored expressions

    Arguments
    ---------
    ode : gotran.ODE
        The finalized ODE for which the monitored expression should be computed
    monitored : tuple, list
        A tuple/list of strings containing the name of the monitored expressions
    function_name : str
        The name of the function which should be generated
    result_name : str
        The name of the variable storing the rhs result
    params : dict
        Parameters determining how the code should be generated
    """

    check_arg(ode, ODE)
    if not ode.is_finalized:
        error(
            "Cannot compute right hand side expressions if the ODE is " "not finalized",
        )

    check_arg(monitored, (tuple, list), itemtypes=str)
    monitored_exprs = []
    for expr_str in monitored:
        obj = ode.present_ode_objects.get(expr_str)
        if not isinstance(obj, Expression):
            error(f"{expr_str} is not an expression in the {ode} ODE")

        monitored_exprs.append(obj)

    descr = f"Computes monitored expressions of the {ode} ODE"
    return CodeComponent(
        "MonitoredExpressions",
        ode,
        function_name,
        descr,
        params=params,
        **{result_name: monitored_exprs},
    )


def componentwise_derivative(ode, indices, params=None, result_name="dy"):
    """
    Return an ODEComponent holding the expressions for the ith
    state derivative

    Arguments
    ---------
    ode : gotran.ODE
        The finalized ODE for which the ith derivative should be computed
    indices : int, list of ints
        The index
    params : dict
        Parameters determining how the code should be generated
    result_name : str
        The name of the result variable
    """
    check_arg(ode, ODE)
    if not ode.is_finalized:
        error("Cannot compute component wise derivatives if ODE is " "not finalized")
    indices = listwrap(indices)

    check_arg(indices, list, itemtypes=int)
    check_arg(result_name, str)

    if len(indices) == 0:
        error("expected at least on index")

    registered = []
    for index in indices:
        if index < 0 or index >= ode.num_full_states:
            error(
                "Expected the passed indices to be between 0 and the "
                "number of states in the ode, got {0}.".format(index),
            )
        if index in registered:
            error(f"Index {index} appeared twice.")

        registered.append(index)

    # Get state expression
    exprs = [ode.state_expressions[index] for index in indices]

    results = {result_name: exprs}

    return CodeComponent(
        f"componentwise_derivatives_{'_'.join(expr.state.name for expr in exprs)}",
        ode,
        "",
        "",
        params=params,
        **results,
    )


def linearized_derivatives(
    ode,
    function_name="linear_derivatives",
    result_names=["linearized", "dy"],
    only_linear=True,
    include_rhs=False,
    nonlinear_last=False,
    params=None,
):
    """
    Return an ODEComponent holding the linearized derivative expressions

    Arguments
    ---------
    ode : gotran.ODE
        The ODE for which derivatives should be linearized
    function_name : str
        The name of the function which should be generated
    result_names : str
        The name of the variable storing the linearized derivatives and the
        rhs evaluation if that is included.
    only_linear : bool
        If True, only linear terms will be linearized
    include_rhs : bool
        If True, rhs evaluation will be included in the generated code.
    nonlinear_last : bool
        If True the nonlinear expressions are added last after a comment
    params : dict
        Parameters determining how the code should be generated
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return LinearizedDerivativeComponent(
        ode,
        function_name,
        result_names,
        only_linear,
        include_rhs,
        nonlinear_last,
        params,
    )


def jacobian_expressions(
    ode,
    function_name="compute_jacobian",
    result_name="jac",
    params=None,
):
    """
    Return an ODEComponent holding expressions for the jacobian

    Arguments
    ---------
    ode : gotran.ODE
        The ODE for which the jacobian expressions should be computed
    function_name : str
        The name of the function which should be generated
    result_name : str
        The name of the variable storing the jacobian result
    params : dict
        Parameters determining how the code should be generated
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return JacobianComponent(
        ode,
        function_name=function_name,
        result_name=result_name,
        params=params,
    )


def jacobian_action_expressions(
    jacobian,
    with_body=True,
    function_name="compute_jacobian_action",
    result_name="jac_action",
    params=None,
):
    """
    Return an ODEComponent holding expressions for the jacobian action

    Arguments
    ---------
    jacobian : gotran.JacobianComponent
        The ODEComponent holding expressions for the jacobian
    with_body : bool
        If true, the body for computing the jacobian will be included
    function_name : str
        The name of the function which should be generated
    result_name : str
        The name of the variable storing the jacobian diagonal result
    params : dict
        Parameters determining how the code should be generated
    """

    check_arg(jacobian, JacobianComponent)
    return JacobianActionComponent(
        jacobian,
        with_body,
        function_name,
        result_name,
        params=params,
    )


def diagonal_jacobian_expressions(
    jacobian,
    function_name="compute_diagonal_jacobian",
    result_name="diag_jac",
    params=None,
):
    """
    Return an ODEComponent holding expressions for the diagonal jacobian

    Arguments
    ---------
    jacobian : gotran.JacobianComponent
        The Jacobian of the ODE
    function_name : str
        The name of the function which should be generated
    result_name : str
        The name of the variable storing the jacobian diagonal result
    params : dict
        Parameters determining how the code should be generated
    """
    return DiagonalJacobianComponent(
        jacobian,
        function_name,
        result_name,
        params=params,
    )


def diagonal_jacobian_action_expressions(
    diagonal_jacobian,
    with_body=True,
    function_name="compute_diagonal_jacobian_action",
    result_name="diag_jac_action",
    params=None,
):
    """
    Return an ODEComponent holding expressions for the diagonal jacobian action

    Arguments
    ---------
    diagonal_jacobian : gotran.DiagonalJacobianComponent
        The ODEComponent holding expressions for the diagonal jacobian
    with_body : bool
        If true, the body for computing the jacobian will be included
    function_name : str
        The name of the function which should be generated
    result_name : str
        The name of the variable storing the jacobian diagonal result
    params : dict
        Parameters determining how the code should be generated
    """

    check_arg(diagonal_jacobian, DiagonalJacobianComponent)
    return DiagonalJacobianActionComponent(
        diagonal_jacobian,
        with_body,
        function_name,
        result_name,
        params=params,
    )


def factorized_jacobian_expressions(
    jacobian,
    function_name="lu_factorize",
    params=None,
):
    """
    Return an ODEComponent holding expressions for the factorized jacobian

    Arguments
    ---------
    jacobian : gotran.JacobianComponent
        The ODEComponent holding expressions for the jacobian
    params : dict
        Parameters determining how the code should be generated
    """
    check_arg(jacobian, JacobianComponent)
    return FactorizedJacobianComponent(jacobian, function_name, params=params)


def forward_backward_subst_expressions(
    factorized,
    function_name="forward_backward_subst",
    result_name="dx",
    residual_name="F",
    params=None,
):
    """
    Return an ODEComponent holding expressions for the forward backward
    substitions for a factorized jacobian

    Arguments
    ---------
    factorized : gotran.FactorizedJacobianComponent
        The ODEComponent holding expressions for the factorized jacobian
    function_name : str
        The name of the function which should be generated
    result_name : str
        The name of the result (increment)
    residual_name : str
        The name of the residual
    params : dict
        Parameters determining how the code should be generated
    """
    check_arg(factorized, FactorizedJacobianComponent)
    return ForwardBackwardSubstitutionComponent(
        factorized,
        function_name=function_name,
        result_name=result_name,
        residual_name=residual_name,
        params=params,
    )


class JacobianComponent(CodeComponent):
    """
    An ODEComponent which keeps all expressions for the Jacobian of the rhs
    """

    def __init__(
        self,
        ode,
        function_name="compute_jacobian",
        result_name="jac",
        params=None,
    ):
        """
        Create a JacobianComponent

        Arguments
        ---------
        ode : gotran.ODE
            The parent component of this ODEComponent
        function_name : str
            The name of the function which should be generated
        result_name : str
            The name of the variable storing the jacobian result
        params : dict
            Parameters determining how the code should be generated
        """
        check_arg(ode, ODE)

        # Call base class using empty result_expressions
        descr = f"Compute the jacobian of the right hand side of the {ode} ODE"
        super(JacobianComponent, self).__init__(
            "Jacobian",
            ode,
            function_name,
            descr,
            params=params,
        )
        check_arg(result_name, str)

        timer = Timer("Computing jacobian")  # noqa:F841

        # Gather state expressions and states
        state_exprs = self.root.state_expressions
        states = self.root.full_states

        # Create Jacobian matrix
        N = len(states)
        self.jacobian = sp.Matrix(N, N, lambda i, j: 0.0)

        self.num_nonzero = 0

        self.shapes[result_name] = (N, N)

        state_dict = dict((state.sym, ind) for ind, state in enumerate(states))
        time_sym = states[0].time.sym

        might_take_time = N >= 10

        if might_take_time:
            info(f"Calculating Jacobian of {ode.name}. Might take some time...")
            sys.stdout.flush()

        for i, expr in enumerate(state_exprs):

            states_syms = sorted(
                (state_dict[sym], sym)
                for sym in ode_primitives(expr.expr, time_sym)
                if sym in state_dict
            )

            self.add_comment(
                f"Expressions for the sparse jacobian of state {expr.state.name}",
                dependent=expr,
            )

            for j, sym in states_syms:
                time_diff = Timer("Differentiate state_expressions")
                jac_ij = expr.expr.diff(sym)
                del time_diff
                self.num_nonzero += 1
                jac_ij = self.add_indexed_expression(
                    result_name,
                    (i, j),
                    jac_ij,
                    dependent=expr,
                )

                self.jacobian[i, j] = jac_ij

        if might_take_time:
            info(" done")

        # Call recreate body with the jacobian expressions as the result
        # expressions
        results = {result_name: self.indexed_objects(result_name)}
        results, body_expressions = self._body_from_results(**results)
        self.body_expressions = self._recreate_body(body_expressions, **results)


class DiagonalJacobianComponent(CodeComponent):
    """
    An ODEComponent which keeps all expressions for the Jacobian of the rhs
    """

    def __init__(
        self,
        jacobian,
        function_name="compute_diagonal_jacobian",
        result_name="diag_jac",
        params=None,
    ):
        """
        Create a DiagonalJacobianComponent

        Arguments
        ---------
        jacobian : gotran.JacobianComponent
            The Jacobian of the ODE
        function_name : str
            The name of the function which should be generated
        result_name : str (optional)
            The basename of the indexed result expression
        params : dict
            Parameters determining how the code should be generated
        """
        check_arg(jacobian, JacobianComponent)

        descr = (
            "Compute the diagonal jacobian of the right hand side of the "
            "{0} ODE".format(jacobian.root)
        )
        super(DiagonalJacobianComponent, self).__init__(
            "DiagonalJacobian",
            jacobian.root,
            function_name,
            descr,
            params=params,
        )

        what = "Computing diagonal jacobian"
        timer = Timer(what)  # noqa: F841

        self.add_comment(what)

        N = jacobian.jacobian.shape[0]
        self.shapes[result_name] = (N,)
        jacobian_name = jacobian.results[0]

        # Create IndexExpressions of the diagonal Jacobian
        for expr in jacobian.indexed_objects(jacobian_name):
            if expr.indices[0] == expr.indices[1]:
                self.add_indexed_expression(result_name, expr.indices[0], expr.expr)

        self.diagonal_jacobian = sp.Matrix(N, N, lambda i, j: 0.0)

        for i in range(N):
            self.diagonal_jacobian[i, i] = jacobian.jacobian[i, i]

        # Call recreate body with the jacobian diagonal expressions as the
        # result expressions
        results = {result_name: self.indexed_objects(result_name)}
        results, body_expressions = self._body_from_results(**results)
        self.body_expressions = self._recreate_body(body_expressions, **results)


class JacobianActionComponent(CodeComponent):
    """
    Jacobian action component which returns the expressions for Jac*x
    """

    def __init__(
        self,
        jacobian,
        with_body=True,
        function_name="compute_jacobian_action",
        result_name="jac_action",
        params=None,
    ):
        """
        Create a JacobianActionComponent

        Arguments
        ---------
        jacobian : gotran.JacobianComponent
            The Jacobian of the ODE
        with_body : bool
            If true, the body for computing the jacobian will be included
        function_name : str
            The name of the function which should be generated
        result_name : str
            The basename of the indexed result expression
        params : dict
            Parameters determining how the code should be generated
        """
        timer = Timer("Computing jacobian action component")  # noqa: F841
        check_arg(jacobian, JacobianComponent)
        descr = (
            "Compute the jacobian action of the right hand side of the "
            "{0} ODE".format(jacobian.root)
        )
        super(JacobianActionComponent, self).__init__(
            "JacobianAction",
            jacobian.root,
            function_name,
            descr,
            params=params,
        )

        x = self.root.full_state_vector
        jac = jacobian.jacobian
        jacobian_name = jacobian.results[0]

        # Create Jacobian action vector
        self.action_vector = sp.Matrix(len(x), 1, lambda i, j: 0)

        self.add_comment("Computing the action of the jacobian")

        self.shapes[result_name] = (len(x),)
        self.shapes[jacobian_name] = jacobian.shapes[jacobian_name]
        for i, expr in enumerate(jac * x):
            self.action_vector[i] = self.add_indexed_expression(result_name, i, expr)

        # Call recreate body with the jacobian action expressions as the
        # result expressions
        results = {result_name: self.indexed_objects(result_name)}
        if with_body:
            results, body_expressions = self._body_from_results(**results)
        else:
            body_expressions = results[result_name]

        self.body_expressions = self._recreate_body(body_expressions, **results)


class DiagonalJacobianActionComponent(CodeComponent):
    """
    Jacobian action component which returns the expressions for Jac*x
    """

    def __init__(
        self,
        diagonal_jacobian,
        with_body=True,
        function_name="compute_diagonal_jacobian_action",
        result_name="diag_jac_action",
        params=None,
    ):
        """
        Create a DiagonalJacobianActionComponent

        Arguments
        ---------
        jacobian : gotran.JacobianComponent
            The Jacobian of the ODE
        with_body : bool
            If true, the body for computing the jacobian will be included
        function_name : str
            The name of the function which should be generated
        result_name : str
            The basename of the indexed result expression
        params : dict
            Parameters determining how the code should be generated
        """
        timer = Timer("Computing jacobian action component")  # noqa: F841
        check_arg(diagonal_jacobian, DiagonalJacobianComponent)
        descr = (
            "Compute the diagonal jacobian action of the right hand side "
            "of the {0} ODE".format(diagonal_jacobian.root)
        )
        super(DiagonalJacobianActionComponent, self).__init__(
            "DiagonalJacobianAction",
            diagonal_jacobian.root,
            function_name,
            descr,
            params=params,
        )

        x = self.root.full_state_vector
        jac = diagonal_jacobian.diagonal_jacobian

        self._action_vector = sp.Matrix(len(x), 1, lambda i, j: 0)

        self.add_comment("Computing the action of the jacobian")

        # Create Jacobian matrix
        self.shapes[result_name] = (len(x),)
        for i, expr in enumerate(jac * x):
            self._action_vector[i] = self.add_indexed_expression(result_name, i, expr)

        # Call recreate body with the jacobian action expressions as the
        # result expressions
        results = {result_name: self.indexed_objects(result_name)}
        if with_body:
            results, body_expressions = self._body_from_results(**results)
        else:
            body_expressions = results[result_name]

        self.body_expressions = self._recreate_body(body_expressions, **results)


class FactorizedJacobianComponent(CodeComponent):
    """
    Class to generate expressions for symbolicaly factorizing a jacobian
    """

    def __init__(self, jacobian, function_name="lu_factorize", params=None):
        """
        Create a FactorizedJacobianComponent

        Arguments
        ---------
        jacobian : gotran.JacobianComponent
            The Jacobian of the ODE
        function_name : str
            The name of the function which should be generated
        params : dict
            Parameters determining how the code should be generated
        """

        timer = Timer("Computing factorization of jacobian")  # noqa: F841
        check_arg(jacobian, JacobianComponent)
        descr = f"Symbolically factorize the jacobian of the {jacobian.root} ODE"
        super(FactorizedJacobianComponent, self).__init__(
            "FactorizedJacobian",
            jacobian.root,
            function_name,
            descr,
            params=params,
            use_default_arguments=False,
            additional_arguments=jacobian.results,
        )

        self.add_comment(f"Factorizing jacobian of {self.root.name}")

        jacobian_name = jacobian.results[0]

        # Recreate jacobian using only sympy Symbols
        jac_orig = jacobian.jacobian

        # Size of system
        n = jac_orig.rows
        jac = sp.Matrix(n, n, lambda i, j: sp.S.Zero)

        for i in range(n):
            for j in range(n):
                # print jac_orig[i,j]
                if not jac_orig[i, j].is_zero:
                    name = sympycode(jac_orig[i, j])
                    jac[i, j] = sp.Symbol(
                        name,
                        real=True,
                        imaginary=False,
                        commutative=True,
                        hermitian=True,
                        complex=True,
                    )
                    print(jac[i, j])
        p = []

        self.shapes[jacobian_name] = (n, n)

        def add_intermediate_if_changed(jac, jac_ij, i, j):
            # If item has changed
            if jac_ij != jac[i, j]:
                print("jac", i, j, jac_ij)
                jac[i, j] = self.add_indexed_expression(jacobian_name, (i, j), jac_ij)

        # Do the factorization
        for j in range(n):

            for i in range(j):

                # Get sympy expr of A_ij
                jac_ij = jac[i, j]

                # Build sympy expression
                for k in range(i):
                    jac_ij -= jac[i, k] * jac[k, j]

                add_intermediate_if_changed(jac, jac_ij, i, j)

            pivot = -1

            for i in range(j, n):

                # Get sympy expr of A_ij
                jac_ij = jac[i, j]

                # Build sympy expression
                for k in range(j):
                    jac_ij -= jac[i, k] * jac[k, j]

                add_intermediate_if_changed(jac, jac_ij, i, j)

                # find the first non-zero pivot, includes any expression
                if pivot == -1 and jac[i, j]:
                    pivot = i

            if pivot < 0:
                # this result is based on iszerofunc's analysis of the
                # possible pivots, so even though the element may not be
                # strictly zero, the supplied iszerofunc's evaluation gave
                # True
                error("No nonzero pivot found; symbolic inversion failed.")

            if pivot != j:  # row must be swapped
                jac.row_swap(pivot, j)
                p.append([pivot, j])
                print("Pivoting!!")

            # Scale with diagonal
            if not jac[j, j]:
                error("Diagonal element of the jacobian is zero. " "Inversion failed")

            scale = 1 / jac[j, j]
            for i in range(j + 1, n):

                # Get sympy expr of A_ij
                jac_ij = jac[i, j]
                jac_ij *= scale
                add_intermediate_if_changed(jac, jac_ij, i, j)

        # Store factorized jacobian
        self.factorized_jacobian = jac
        self.num_nonzero = sum(
            not jac[i, j].is_zero for i in range(n) for j in range(n)
        )

        # No need to call recreate body expressions
        self.body_expressions = self.ode_objects

        self.used_states = set()
        self.used_parameters = set()


class ForwardBackwardSubstitutionComponent(CodeComponent):
    """
    Class to generate a forward backward substiution algorithm for
    symbolically factorized jacobian
    """

    def __init__(
        self,
        factorized,
        function_name="forward_backward_subst",
        result_name="dx",
        residual_name="F",
        params=None,
    ):
        """
        Create a JacobianForwardBackwardSubstComponent

        Arguments
        ---------
        factorized : gotran.FactorizedJacobianComponent
            The factorized jacobian of the ODE
        function_name : str
            The name of the function which should be generated
        result_name : str
            The name of the result (increment)
        residual_name : str
            The name of the residual
        params : dict
            Parameters determining how the code should be generated
        """
        timer = Timer("Computing forward backward substituion component")  # noqa: F841
        check_arg(factorized, FactorizedJacobianComponent)
        jacobian_name = list(factorized.shapes.keys())[0]
        descr = (
            "Symbolically forward backward substitute linear system "
            "of {0} ODE".format(factorized.root)
        )
        super(ForwardBackwardSubstitutionComponent, self).__init__(
            "ForwardBackwardSubst",
            factorized.root,
            function_name,
            descr,
            params=params,
            use_default_arguments=False,
            additional_arguments=[residual_name],
        )

        self.add_comment(
            f"Forward backward substituting factorized linear system {self.root.name}",
        )

        # Recreate jacobian using only sympy Symbols
        jac_orig = factorized.factorized_jacobian

        # Size of system
        n = jac_orig.rows
        jac = sp.Matrix(n, n, lambda i, j: sp.S.Zero)

        for i in range(n):
            for j in range(n):
                # print jac_orig[i,j]
                if not jac_orig[i, j].is_zero:
                    name = sympycode(jac_orig[i, j])
                    jac[i, j] = sp.Symbol(
                        name,
                        real=True,
                        imaginary=False,
                        commutative=True,
                        hermitian=True,
                        complex=True,
                    )
                    print(jac[i, j])

        self.shapes[jacobian_name] = (n, n)
        self.shapes[residual_name] = (n,)
        self.shapes[result_name] = (n,)

        F = []
        dx = []
        # forward substitution, all diag entries are scaled to 1
        for i in range(n):

            F.append(self.add_indexed_object(residual_name, i))
            dx.append(self.add_indexed_expression(result_name, i, F[i]))

            for j in range(i):
                if jac[i, j].is_zero:
                    continue
                dx[i] = self.add_indexed_expression(
                    result_name,
                    i,
                    dx[i] - dx[j] * jac[i, j],
                )

        # backward substitution
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if jac[i, j].is_zero:
                    continue
                dx[i] = self.add_indexed_expression(
                    result_name,
                    i,
                    dx[i] - dx[j] * jac[i, j],
                )

            dx[i] = self.add_indexed_expression(result_name, i, dx[i] / jac[i, i])

        # No need to call recreate body expressions
        self.body_expressions = [
            obj
            for obj in self.ode_objects
            if isinstance(obj, (IndexedExpression, Comment))
        ]

        self.results = [result_name]
        self.used_states = set()
        self.used_parameters = set()


class LinearizedDerivativeComponent(CodeComponent):
    """
    A component for all linear and linearized derivatives
    """

    def __init__(
        self,
        ode,
        function_name="linear_derivatives",
        result_names=["linearized", "rhs"],
        only_linear=True,
        include_rhs=False,
        nonlinear_last=False,
        params=None,
    ):
        """
        Return an ODEComponent holding the linearized derivative expressions

        Arguments
        ---------
        ode : gotran.ODE
            The ODE for which derivatives should be linearized
        function_name : str
            The name of the function which should be generated
        result_names : str
            The name of the variable storing the linearized derivatives and the
            rhs evaluation if that is included.
        only_linear : bool
            If True, only linear terms will be linearized
        include_rhs : bool
            If True, rhs evaluation will be included in the generated code.
        params : dict
            Parameters determining how the code should be generated
        """

        check_kwarg(result_names, "result_names", list, itemtypes=str)
        if len(result_names) != 2:
            error("expected the length of 'result_names' to be 2")

        descr = "Computes the linearized derivatives for all linear derivatives"
        super(LinearizedDerivativeComponent, self).__init__(
            "LinearizedDerivatives",
            ode,
            function_name,
            descr,
            params=params,
        )

        check_arg(ode, ODE)
        assert ode.is_finalized

        self.linear_derivative_indices = [0] * self.root.num_full_states
        linearized_name, rhs_name = result_names

        state_exprs = self.root.state_expressions
        self.shapes[linearized_name] = (self.root.num_full_states,)
        if include_rhs:
            self.shapes[rhs_name] = (self.root.num_full_states,)

        nonlinear_exprs = []
        for ind, expr in enumerate(state_exprs):

            expr_diff = expr.expr.diff(expr.state.sym)

            if expr_diff and expr.state.sym not in expr_diff.args:
                self.linear_derivative_indices[ind] = 1
            elif only_linear:
                continue

            # Append nonlinear expressions for later addition
            if nonlinear_last and self.linear_derivative_indices[ind] == 0:
                nonlinear_exprs.append((linearized_name, ind, expr_diff))
            else:
                self.add_indexed_expression(
                    linearized_name,
                    ind,
                    expr_diff,
                    dependent=expr,
                )

        if nonlinear_exprs:
            self.add_comment("Nonlinear linearized expressions", dependent=expr)
            for linearized_name, ind, expr_diff in nonlinear_exprs:
                self.add_indexed_expression(
                    linearized_name,
                    ind,
                    expr_diff,
                    dependent=expr,
                )

        # Call recreate body with the jacobian action expressions as the
        # result expressions
        results = {linearized_name: self.indexed_objects(linearized_name)}
        if include_rhs:
            results[rhs_name] = state_exprs
        results, body_expressions = self._body_from_results(**results)
        self.body_expressions = self._recreate_body(body_expressions, **results)


class CommonSubExpressionODE(ODE):
    """
    Class which flattens the component structue of an ODE to just one.
    It uses common sub expressions as intermediates to reduce complexity
    of the derivative expressions.
    """

    def __init__(self, ode):
        check_arg(ode, ODE)
        assert ode.is_finalized

        timer = Timer("Extract common sub expressions")  # noqa: F841

        newname = ode.name + "_CSE"

        # Call super class
        super(CommonSubExpressionODE, self).__init__(newname, ode.ns)

        # Add states and parameters
        atoms = []
        for state in ode.full_states:
            atoms.append(self.add_state(state.name, state.param))

        for param in ode.parameters:
            atoms.append(self.add_parameter(param.name, param.param))

        # Collect all expanded state expressions
        org_state_expressions = ode.state_expressions
        expanded_state_exprs = [
            ode.expanded_expressions[obj.name] for obj in org_state_expressions
        ]

        # Call sympy common sub expression reduction
        cse_exprs, cse_state_exprs = cse(
            expanded_state_exprs,
            symbols=sp.numbered_symbols("cse_"),
            optimizations=[],
        )
        cse_cnt = 0
        cse_subs = {}

        # Register the common sub expressions as Intermediates
        for sub, expr in cse_exprs:

            # If the expression is just one of the atoms of the ODE we skip
            # the cse expressions but add a subs for the atom
            if expr in atoms:
                cse_subs[sub] = expr
            else:
                cse_subs[sub] = self.add_intermediate(
                    f"cse_{cse_cnt}",
                    expr.xreplace(cse_subs),
                )
                cse_cnt += 1

        # Register the state expressions
        for org_state_expr, state_expr in zip(org_state_expressions, cse_state_exprs):

            exp_expr = state_expr.xreplace(cse_subs)
            state = self.get_object(org_state_expr.state.name)[1]

            # If state derivative
            if isinstance(org_state_expr, StateDerivative):
                self.add_derivative(state, state.time.sym, exp_expr)

            # If algebraic
            elif isinstance(org_state_expr, AlgebraicExpression):
                self.add_algebraic(state, exp_expr)

            else:
                error("Should not come here!")

        self.finalize()
