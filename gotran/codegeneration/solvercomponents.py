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
    "ExplicitEuler",
    "explicit_euler_solver",
    "RushLarsen",
    "rush_larsen_solver",
    "GeneralizedRushLarsen",
    "generalized_rush_larsen_solver",
    "HybridGeneralizedRushLarsen",
    "hybrid_generalized_rush_larsen_solver",
    "SimplifiedImplicitEuler",
    "simplified_implicit_euler_solver",
    "get_solver_fn",
]

# System imports
import sys

from modelparameters.logger import debug, error, info

# ModelParameters imports
from modelparameters.sympytools import Conditional, sp

# Local imports
from modelparameters.utils import Timer, check_arg

from ..model.ode import ODE
from .codecomponent import CodeComponent


def explicit_euler_solver(ode, function_name="forward_explicit_euler", params=None):
    """
    Return an ODEComponent holding expressions for the explicit Euler method

    Arguments
    ---------
    ode : gotran.ODE
        The ODE for which the jacobian expressions should be computed
    function_name : str
        The name of the function which should be generated
    params : dict
        Parameters determining how the code should be generated
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return ExplicitEuler(ode, function_name=function_name, params=params)


def rush_larsen_solver(
    ode,
    function_name="forward_rush_larsen",
    delta=1e-8,
    params=None,
):
    """
    Return an ODEComponent holding expressions for the Rush-Larsen method

    Arguments
    ---------
    ode : gotran.ODE
        The ODE for which the jacobian expressions should be computed
    function_name : str
        The name of the function which should be generated
    delta : float
        Value to safeguard the evaluation of the Rush-Larsen step.
    params : dict
        Parameters determining how the code should be generated
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return RushLarsen(ode, function_name=function_name, delta=delta, params=params)


def generalized_rush_larsen_solver(
    ode,
    function_name="forward_generalized_rush_larsen",
    delta=1e-8,
    params=None,
):
    """
    Return an ODEComponent holding expressions for the generalized Rush-Larsen method

    Arguments
    ---------
    ode : gotran.ODE
        The ODE for which the jacobian expressions should be computed
    function_name : str
        The name of the function which should be generated
    delta : float
        Value to safeguard the evaluation of the Rush-Larsen step.
    params : dict
        Parameters determining how the code should be generated
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return GeneralizedRushLarsen(
        ode,
        function_name=function_name,
        delta=delta,
        params=params,
    )


def hybrid_generalized_rush_larsen_solver(
    ode,
    function_name="forward_hybrid_generalized_rush_larsen",
    delta=1e-8,
    params=None,
    stiff_states=None,
):
    """
    Return an ODEComponent holding expressions for the generalized Rush-Larsen method

    Arguments
    ---------
    ode : gotran.ODE
        The ODE for which the jacobian expressions should be computed
    function_name : str
        The name of the function which should be generated
    delta : float
        Value to safeguard the evaluation of the Rush-Larsen step.
    params : dict
        Parameters determining how the code should be generated
    stiff_state_variables : list of str
        States that are stiff and should be solved with GRL1. The remaining states are solved with explicit euler
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return HybridGeneralizedRushLarsen(
        ode,
        function_name=function_name,
        delta=delta,
        params=params,
        stiff_state_variables=stiff_states,
    )


def simplified_implicit_euler_solver(
    ode,
    function_name="forward_simplified_implicit_euler",
    numeric_jacobian=False,
    params=None,
):
    """
    Return an ODEComponent holding expressions for the simplified
    implicit Euler method

    Arguments
    ---------
    ode : gotran.ODE
        The ODE for which the jacobian expressions should be computed
    function_name : str
        The name of the function which should be generated
    numeric_jacobian : bool
        If True use numeric calculated diagonal jacobian.
    params : dict
        Parameters determining how the code should be generated
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return SimplifiedImplicitEuler(
        ode,
        function_name=function_name,
        numeric_jacobian=numeric_jacobian,
        params=params,
    )


def get_solver_fn(solver_type):
    return {
        "explicit_euler": explicit_euler_solver,
        "rush_larsen": rush_larsen_solver,
        "generalized_rush_larsen": generalized_rush_larsen_solver,
        "hybrid_generalized_rush_larsen": hybrid_generalized_rush_larsen_solver,
        "simplified_implicit_euler": simplified_implicit_euler_solver,
    }[solver_type]


class ExplicitEuler(CodeComponent):
    """
    An ODEComponent which compute one step of the explicit Euler scheme
    """

    def __init__(self, ode, function_name="forward_explicit_euler", params=None):
        """
        Create an ExplicitEuler CodeComponent

        Arguments
        ---------
        ode : gotran.ODE
            The parent component of this ODEComponent
        function_name : str
            The name of the function which should be generated
        params : dict
            Parameters determining how the code should be generated
        """
        check_arg(ode, ODE)

        if ode.is_dae:
            error("Cannot generate an explicit Euler forward step for a DAE.")

        # Call base class using empty result_expressions
        descr = (
            f"Compute a forward step using the explicit Euler scheme to the {ode} ODE"
        )
        super(ExplicitEuler, self).__init__(
            "ExplicitEuler",
            ode,
            function_name,
            descr,
            params=params,
            additional_arguments=["dt"],
        )

        # Recount the expressions if representation of states are "array" as
        # then the method is not full explicit
        recount = self._params.states.representation != "array"

        # Gather state expressions and states
        state_exprs = self.root.state_expressions
        states = self.root.full_states
        result_name = self._params.states.array_name
        self.shapes[result_name] = (len(states),)

        # Get time step and start creating the update algorithm
        if self._params.states.add_offset:
            offset_str = f"{result_name}_offset"
        else:
            offset_str = ""

        dt = self.root._dt.sym
        for i, expr in enumerate(state_exprs):
            dependent = expr if recount else None
            self.add_indexed_expression(
                result_name,
                (i,),
                expr.state.sym + dt * expr.sym,
                offset_str,
                dependent=dependent,
                enum=expr.state,
            )

        # Call recreate body with the solver expressions as the result
        # expressions
        results = {result_name: self.indexed_objects(result_name)}
        results, body_expressions = self._body_from_results(**results)
        self.body_expressions = self._recreate_body(body_expressions, **results)


def fraction_numerator_is_nonzero(expr):
    """Perform a very cheap check to detect if a fraction is definitely non-zero."""
    expr_type = type(expr)
    if expr_type == sp.Pow:
        # check if the expression is on the form a**-1
        a, b = expr.args
        if type(b) is sp.numbers.NegativeOne:
            return True
        else:
            # we won't do any further checks
            return False
    elif expr_type == sp.Mul:
        # check if all factors are non-zero
        args = expr.args
        certainly_nonzero_args = []
        potentially_nonzero_args = []
        for e in args:
            if len(e.free_symbols) == 0 and e.is_nonzero:
                certainly_nonzero_args.append(e)
            else:
                potentially_nonzero_args.append(e)

        if len(potentially_nonzero_args) == 0:
            # all factors are certainly nonzero
            return True

        # check all potentially non-zero factors
        for e in potentially_nonzero_args:
            if not fraction_numerator_is_nonzero(e):
                return False
        else:
            return True
    else:
        return False


class RushLarsen(CodeComponent):
    """
    An ODEComponent which compute one step of the Rush-Larsen scheme
    """

    def __init__(
        self,
        ode,
        function_name="forward_rush_larsen",
        delta=1e-8,
        params=None,
    ):
        """
        Create a RushLarsen Solver component

        Arguments
        ---------
        ode : gotran.ODE
            The parent component of this ODEComponent
        function_name : str
            The name of the function which should be generated
        delta : float
            Value to safeguard the evaluation of the Rush-Larsen step.
        params : dict
            Parameters determining how the code should be generated
        """

        timer = Timer("Create RushLarsen expressions")
        check_arg(ode, ODE)

        if ode.is_dae:
            error("Cannot generate a Rush-Larsen forward step for a DAE.")

        # Call base class using empty result_expressions
        descr = f"Compute a forward step using the Rush-Larsen scheme to the {ode} ODE"
        super(RushLarsen, self).__init__(
            "RushLarsen",
            ode,
            function_name,
            descr,
            params=params,
            additional_arguments=["dt"],
        )

        # Recount the expressions if representation of states are "array" as
        # then the method is not full explcit
        recount = self._params.states.representation != "array"

        # Gather state expressions and states
        state_exprs = self.root.state_expressions
        states = self.root.full_states

        result_name = self._params.states.array_name
        self.shapes[result_name] = (len(states),)

        # Get time step and start creating the update algorithm
        if self._params.states.add_offset:
            offset_str = f"{result_name}_offset"
        else:
            offset_str = ""

        might_take_time = len(states) >= 10

        if might_take_time:
            info(f"Calculating derivatives of {ode.name}. Might take some time...")
            sys.stdout.flush()

        dt = self.root._dt.sym
        for i, expr in enumerate(state_exprs):

            dependent = expr if recount else None

            # Diagonal jacobian value
            time_diff = Timer("Differentiate state_expressions for RushLarsen")
            expr_diff = expr.expr.diff(expr.state.sym)
            del time_diff

            # print expr.state.sym, expr_diff, expr_diff.args
            if expr_diff and expr.state.sym not in expr_diff.args:

                linearized_name = expr.name + "_linearized"
                linearized = self.add_intermediate(
                    linearized_name,
                    expr_diff,
                    dependent=dependent,
                )

                need_zero_div_check = not fraction_numerator_is_nonzero(expr_diff)
                if not need_zero_div_check:
                    debug(
                        f"{linearized_name} cannot be zero. Skipping zero division check",
                    )

                RL_term = expr.sym / linearized * (sp.exp(linearized * dt) - 1)
                if need_zero_div_check:
                    RL_term = Conditional(
                        abs(linearized) > delta,
                        RL_term,
                        dt * expr.sym,
                    )

                # Solve "exact" using exp
                self.add_indexed_expression(
                    result_name,
                    (i,),
                    expr.state.sym + RL_term,
                    offset_str,
                    dependent=dependent,
                    enum=expr.state,
                )

            else:

                # Explicit Euler step
                self.add_indexed_expression(
                    result_name,
                    (i,),
                    expr.state.sym + dt * expr.sym,
                    offset_str,
                    dependent=dependent,
                    enum=expr.state,
                )

        if might_take_time:
            info(" done")

        # Call recreate body with the solver expressions as the result
        # expressions
        del timer
        results = {result_name: self.indexed_objects(result_name)}
        results, body_expressions = self._body_from_results(**results)
        self.body_expressions = self._recreate_body(body_expressions, **results)


class RushLarsenOneStep(CodeComponent):
    """
    An ODEComponent which compute one step of the Rush-Larsen scheme
    """

    def __init__(self, ode, function_name="forward_rush_larsen", params=None):
        """
        Create a RushLarsen Solver component

        Arguments
        ---------
        ode : gotran.ODE
            The parent component of this ODEComponent
        function_name : str
            The name of the function which should be generated
        params : dict
            Parameters determining how the code should be generated
        """
        check_arg(ode, ODE)

        if ode.is_dae:
            error("Cannot generate a Rush-Larsen forward step for a DAE.")

        # Call base class using empty result_expressions
        descr = f"Compute a forward step using the Rush-Larsen scheme to the {ode} ODE"
        state_name = params.states.array_name
        super(RushLarsen, self).__init__(
            "RushLarsen",
            ode,
            function_name,
            descr,
            params=params,
            additional_arguments=["dt", state_name + "_0"],
        )

        # Recount the expressions if representation of states are "array" as
        # then the method is not full explcit
        recount = self._params.states.representation != "array"

        # Gather state expressions and states
        state_exprs = self.root.state_expressions
        states = self.root.full_states

        result_name = self._params.states.array_name + "_1"
        previous_name = self._params.states.array_name + "_0"
        self.shapes[result_name] = (len(states),)
        self.shapes[previous_name] = (len(states),)

        # Get time step and start creating the update algorithm
        if self._params.states.add_offset:
            offset_str = f"{result_name}_offset"
        else:
            offset_str = ""

        dt = self.root._dt.sym
        for i, expr in enumerate(state_exprs):

            prev = self.add_indexed_object(previous_name, (i,), offset_str)

            dependent = expr if recount else None

            # Diagonal jacobian value
            expr_diff = expr.expr.diff(expr.state.sym)

            if expr_diff and expr.state.sym not in expr_diff:

                linearized = self.add_intermediate(
                    expr.name + "_linearized",
                    expr_diff,
                    dependent=dependent,
                )

                # Solve "exact" using exp
                self.add_indexed_expression(
                    result_name,
                    (i,),
                    prev + expr.sym / linearized * (sp.exp(linearized * dt) - 1),
                    offset_str,
                    dependent=dependent,
                    enum=expr.state,
                )

            else:

                # Explicit Euler step
                self.add_indexed_expression(
                    result_name,
                    (i,),
                    prev + dt * expr.sym,
                    offset_str,
                    dependent=dependent,
                    enum=expr.state,
                )

        # Call recreate body with the solver expressions as the result
        # expressions
        results = {result_name: self.indexed_objects(result_name)}
        results, body_expressions = self._body_from_results(**results)
        self.body_expressions = self._recreate_body(body_expressions, **results)


class GeneralizedRushLarsen(CodeComponent):
    """
    An ODEComponent which compute one step of the Generalized Rush-Larsen (GRL1) scheme
    """

    def __init__(
        self,
        ode,
        function_name="forward_generalized_rush_larsen",
        delta=1e-8,
        params=None,
    ):
        """
        Create a GeneralizedRushLarsen Solver component

        Arguments
        ---------
        ode : gotran.ODE
            The parent component of this ODEComponent
        function_name : str
            The name of the function which should be generated
        delta : float
            Value to safeguard the evaluation of the Rush-Larsen step.
        params : dict
            Parameters determining how the code should be generated
        """
        check_arg(ode, ODE)

        # Call base class using empty result_expressions
        descr = (
            "Compute a forward step using the generalised Rush-Larsen (GRL1) scheme to the "
            "{0} ODE".format(ode)
        )
        super(GeneralizedRushLarsen, self).__init__(
            "GeneralizedRushLarsen",
            ode,
            function_name,
            descr,
            params=params,
            additional_arguments=["dt"],
        )

        # Recount the expressions if representation of states are "array" as
        # then the method is not full explcit
        recount = self._params.states.representation != "array"

        # Gather state expressions and states
        state_exprs = self.root.state_expressions
        states = self.root.full_states
        result_name = self._params.states.array_name

        self.shapes[result_name] = (len(states),)

        # Get time step and start creating the update algorithm
        if self._params.states.add_offset:
            offset_str = f"{result_name}_offset"
        else:
            offset_str = ""

        dt = self.root._dt.sym
        for i, expr in enumerate(state_exprs):

            expr_diff = expr.expr.diff(expr.state.sym)
            dependent = expr if recount else None
            if expr_diff.is_zero:
                self.add_indexed_expression(
                    result_name,
                    (i,),
                    expr.state.sym + dt * expr.sym,
                    offset_str,
                    dependent=dependent,
                    enum=expr.state,
                )
                continue

            linearized_name = expr.name + "_linearized"
            linearized = self.add_intermediate(
                linearized_name,
                expr_diff,
                dependent=dependent,
            )

            need_zero_div_check = not fraction_numerator_is_nonzero(expr_diff)
            if not need_zero_div_check:
                debug(f"{linearized_name} cannot be zero. Skipping zero division check")

            RL_term = expr.sym / linearized * (sp.exp(linearized * dt) - 1)
            if need_zero_div_check:
                RL_term = Conditional(
                    abs(linearized) > delta,
                    RL_term,
                    dt * expr.sym,
                )

            # Solve "exact" using exp
            self.add_indexed_expression(
                result_name,
                (i,),
                expr.state.sym + RL_term,
                offset_str,
                dependent=dependent,
                enum=expr.state,
            )

        # Call recreate body with the solver expressions as the result
        # expressions
        results = {result_name: self.indexed_objects(result_name)}
        results, body_expressions = self._body_from_results(**results)
        self.body_expressions = self._recreate_body(body_expressions, **results)


class HybridGeneralizedRushLarsen(CodeComponent):
    """
    An ODEComponent which compute one step of the hybrid explicit Euler / Generalized Rush-Larsen (GRL1) scheme
    """

    def __init__(
        self,
        ode,
        function_name="forward_hybrid_generalized_rush_larsen",
        delta=1e-8,
        params=None,
        stiff_state_variables=None,
    ):
        """
        Create a HybridGeneralizedRushLarsen Solver component

        Arguments
        ---------
        ode : gotran.ODE
            The parent component of this ODEComponent
        function_name : str
            The name of the function which should be generated
        delta : float
            Value to safeguard the evaluation of the Rush-Larsen step.
        params : dict
            Parameters determining how the code should be generated

        """
        check_arg(ode, ODE)

        # Call base class using empty result_expressions
        descr = f"Compute a forward step using the FE / GRL1 scheme to the {ode} ODE"
        super(HybridGeneralizedRushLarsen, self).__init__(
            "HybridGeneralizedRushLarsen",
            ode,
            function_name,
            descr,
            params=params,
            additional_arguments=["dt"],
        )

        state_names = [s.name for s in self.root.full_states]
        if stiff_state_variables is None:
            stiff_state_variables = []
        elif type(stiff_state_variables) is str:
            stiff_state_variables = stiff_state_variables.split(",")

        for s in stiff_state_variables:
            if s == "":
                continue
            assert s in state_names, f"Unknown state '{s}'"

        # Recount the expressions if representation of states are "array" as
        # then the method is not full explcit
        recount = self._params.states.representation != "array"

        # Gather state expressions and states
        state_exprs = self.root.state_expressions
        states = self.root.full_states
        result_name = self._params.states.array_name

        self.shapes[result_name] = (len(states),)

        # Get time step and start creating the update algorithm
        if self._params.states.add_offset:
            offset_str = f"{result_name}_offset"
        else:
            offset_str = ""

        dt = self.root._dt.sym
        for i, expr in enumerate(state_exprs):
            state_is_stiff = state_names[i] in stiff_state_variables

            expr_diff = expr.expr.diff(expr.state.sym)
            dependent = expr if recount else None
            if not state_is_stiff or expr_diff.is_zero:
                # FE scheme
                self.add_indexed_expression(
                    result_name,
                    (i,),
                    expr.state.sym + dt * expr.sym,
                    offset_str,
                    dependent=dependent,
                    enum=expr.state,
                )
                continue

            linearized_name = expr.name + "_linearized"
            linearized = self.add_intermediate(
                linearized_name,
                expr_diff,
                dependent=dependent,
            )

            need_zero_div_check = not fraction_numerator_is_nonzero(expr_diff)
            if not need_zero_div_check:
                debug(f"{linearized_name} cannot be zero. Skipping zero division check")

            RL_term = expr.sym / linearized * (sp.exp(linearized * dt) - 1)
            if need_zero_div_check:
                RL_term = Conditional(
                    abs(linearized) > delta,
                    RL_term,
                    dt * expr.sym,
                )

            # Solve "exact" using exp
            self.add_indexed_expression(
                result_name,
                (i,),
                expr.state.sym + RL_term,
                offset_str,
                dependent=dependent,
                enum=expr.state,
            )

        # Call recreate body with the solver expressions as the result
        # expressions
        results = {result_name: self.indexed_objects(result_name)}
        results, body_expressions = self._body_from_results(**results)
        self.body_expressions = self._recreate_body(body_expressions, **results)


class SimplifiedImplicitEuler(CodeComponent):
    """
    An ODEComponent which compute one step of a simplified Implicit Euler
    scheme
    """

    def __init__(
        self,
        ode,
        function_name="forward_simplified_implicit_euler",
        numeric_jacobian=False,
        params=None,
    ):
        """
        Create a SimplifiedImplicitEuler CodeComponent

        Arguments
        ---------
        ode : gotran.ODE
            The parent component of this ODEComponent
        function_name : str
            The name of the function which should be generated
        numeric_jacobian : bool
            If True use numeric calculated diagonal jacobian.
        params : dict
            Parameters determining how the code should be generated
        """
        check_arg(ode, ODE)

        if ode.is_dae:
            error("Cannot generate an explicit Euler forward step for a DAE.")

        # Call base class using empty result_expressions
        descr = (
            "Compute a forward step using the simplified implicit Euler"
            "scheme to the {0} ODE".format(ode)
        )
        super(SimplifiedImplicitEuler, self).__init__(
            "SimplifiedImplicitEuler",
            ode,
            function_name,
            descr,
            params=params,
            additional_arguments=["dt"],
        )

        # Recount the expressions if representation of states are "array" as
        # then the method is not full explcit
        recount = self._params.states.representation != "array"

        # Gather state expressions and states
        state_exprs = self.root.state_expressions
        states = self.root.full_states

        result_name = self._params.states.array_name
        self.shapes[result_name] = (len(states),)

        # Get time step and start creating the update algorithm
        if self._params.states.add_offset:
            offset_str = f"{result_name}_offset"
        else:
            offset_str = ""

        dt = self.root._dt.sym
        for i, expr in enumerate(state_exprs):

            dependent = expr if recount else None

            # Diagonal jacobian value
            diag_jac_expr = expr.expr.diff(expr.state.sym)

            if not diag_jac_expr.is_zero:
                diag_jac = self.add_intermediate(
                    expr.name + "_diag_jac",
                    diag_jac_expr,
                    dependent=dependent,
                )
            else:
                diag_jac = 0.0

            # Add simplified single Implicit Euler step
            self.add_indexed_expression(
                result_name,
                (i,),
                expr.state.sym + dt * expr.sym / (1 - dt * diag_jac),
                offset_str,
                enum=expr.state,
            )

        # Call recreate body with the solver expressions as the result
        # expressions
        results = {result_name: self.indexed_objects(result_name)}
        results, body_expressions = self._body_from_results(**results)
        self.body_expressions = self._recreate_body(body_expressions, **results)
