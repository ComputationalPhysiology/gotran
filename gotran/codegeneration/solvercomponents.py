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

__all__ = ["ExplicitEuler", "explicit_euler_solver",
           "RushLarsen", "rush_larsen_solver",
           "SimplifiedImplicitEuler", "simplified_implicit_euler_solver",
           "get_solver_fn"]

# System imports
import sys

# ModelParameters imports
from modelparameters.sympytools import sp
from modelparameters.codegeneration import sympycode

# Local imports
from gotran.common import error, info, debug, check_arg, check_kwarg, \
     scalars, Timer, warning, tuplewrap, parameters, warning
from gotran.common import listwrap
from gotran.model.utils import ode_primitives
from gotran.model.odeobjects import State, Parameter, IndexedObject, Comment
from gotran.model.expressions import *
from gotran.model.ode import ODE
from gotran.codegeneration.codecomponent import CodeComponent

def explicit_euler_solver(ode, function_name="forward_explicit_euler", \
                          params=None):
    """
    Return an ODEComponent holding expressions for the explicit Euler method

    Arguments
    ---------
    ode : ODE
        The ODE for which the jacobian expressions should be computed
    function_name : str
        The name of the function which should be generated
    params : dict
        Parameters determining how the code should be generated
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return ExplicitEuler(ode, function_name=function_name, params=params)

def rush_larsen_solver(ode, function_name="forward_rush_larsen", \
                       params=None):
    """
    Return an ODEComponent holding expressions for the explicit Euler method

    Arguments
    ---------
    ode : ODE
        The ODE for which the jacobian expressions should be computed
    function_name : str
        The name of the function which should be generated
    params : dict
        Parameters determining how the code should be generated
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return RushLarsen(ode, function_name=function_name, params=params)

def simplified_implicit_euler_solver(\
    ode, function_name="forward_simplified_implicit_euler", params=None):
    """
    Return an ODEComponent holding expressions for the simplified
    implicit Euler method

    Arguments
    ---------
    ode : ODE
        The ODE for which the jacobian expressions should be computed
    function_name : str
        The name of the function which should be generated
    params : dict
        Parameters determining how the code should be generated
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return SimplifiedImplicitEuler(ode, function_name=function_name, params=params)

def get_solver_fn(solver_type):
    return {
        'explicit_euler': explicit_euler_solver,
        'rush_larsen': rush_larsen_solver,
        'simplified_implicit_euler': simplified_implicit_euler_solver
    }[solver_type]

class ExplicitEuler(CodeComponent):
    """
    An ODEComponent which compute one step of the explicit Euler algorithm
    """
    def __init__(self, ode, function_name="forward_explicit_euler", \
                 params=None):
        """
        Create an ExplicitEuler 

        Arguments
        ---------
        ode : ODE
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
        descr = "Compute a forward step using the explicit Euler algorithm to the "\
                "{0} ODE".format(ode)
        super(ExplicitEuler, self).__init__("ExplicitEuler", ode, function_name, \
                                            descr, params=params,
                                            additional_arguments=["dt"])

        # Recount the expressions if representation of states are "array" as
        # then the method is not full explcit
        recount = self._params.states.representation != "array"

        # Gather state expressions and states
        state_exprs = self.root.state_expressions
        states = self.root.full_states
        state_dict = dict((state.sym, ind) for ind, state in enumerate(states))
        all_exprs = sorted(self.root.intermediates + self.root.comments \
                           + state_exprs)

        result_name = self._params.states.array_name
        state_offset = self._params.states.add_offset
        self.shapes[result_name] = (len(states),)
        
        self.add_comment("Computing the explicit Euler algorithm")

        # Get time step and start creating the update algorithm
        if self._params.states.add_offset:
            offset_str = "{0}_offset".format(result_name)
        else:
            offset_str = ""
        
        solver_expr_added = False
        total_num_state_expressions = sum(isinstance(expr, StateExpression) \
                                          for expr in all_exprs)
        dt = self.root._dt.sym
        num_state_expressions = 0
        for expr in all_exprs:

            # Check if we are finished
            if num_state_expressions == total_num_state_expressions:
                break
            
            if isinstance(expr, StateExpression):
                num_state_expressions += 1
                i = state_dict[expr.state.sym]
                solver_expr_added = True

                if recount:
                    expr._recount()
                self.add_indexed_expression(result_name, (i,), \
                                            expr.state.sym+dt*expr.sym, offset_str)
                    
            elif solver_expr_added and recount:
                expr._recount()

        # Call recreate body with the jacobian expressions as the result
        # expressions
        results = {result_name:self.indexed_objects(result_name)}
        results, body_expressions = self._body_from_results(**results)
        self.body_expressions = self._recreate_body(\
            body_expressions, **results)

class RushLarsen(CodeComponent):
    """
    An ODEComponent which compute one step of the Rush Larsen algorithm
    """
    def __init__(self, ode, function_name="forward_rush_larsen", \
                 params=None):
        """
        Create a JacobianComponent

        Arguments
        ---------
        ode : ODE
            The parent component of this ODEComponent
        function_name : str
            The name of the function which should be generated
        params : dict
            Parameters determining how the code should be generated
        """
        check_arg(ode, ODE)

        if ode.is_dae:
            error("Cannot generate a Rush Larsen forward step for a DAE.")

        # Call base class using empty result_expressions
        descr = "Compute a forward step using the rush larsen algorithm to the "\
                "{0} ODE".format(ode)
        super(RushLarsen, self).__init__("RushLarsen", ode, function_name, \
                                         descr, params=params,
                                         additional_arguments=["dt"])

        # Recount the expressions if representation of states are "array" as
        # then the method is not full explcit
        recount = self._params.states.representation != "array"

        # Gather state expressions and states
        state_exprs = self.root.state_expressions
        states = self.root.full_states
        state_dict = dict((state.sym, ind) for ind, state in enumerate(states))
        all_exprs = sorted(self.root.intermediates + self.root.comments \
                           + state_exprs)

        result_name = self._params.states.array_name
        state_offset = self._params.states.add_offset
        self.shapes[result_name] = (len(states),)
        
        self.add_comment("Computing the explicit Euler algorithm")

        # Get time step and start creating the update algorithm
        if self._params.states.add_offset:
            offset_str = "{0}_offset".format(result_name)
        else:
            offset_str = ""
        
        solver_expr_added = False
        total_num_state_expressions = sum(isinstance(expr, StateExpression) \
                                          for expr in all_exprs)
        dt = self.root._dt.sym
        num_state_expressions = 0
        for expr in all_exprs:

            # Check if we are finished
            if num_state_expressions == total_num_state_expressions:
                break
            
            if isinstance(expr, StateExpression):
                num_state_expressions += 1
                i = state_dict[expr.state.sym]
                solver_expr_added = True
                expr_diff = expr.expr.diff(expr.state.sym)
                
                if recount:
                    expr._recount()

                if expr_diff and expr.state.sym not in expr_diff:
                    
                    linearized = self.add_intermediate(\
                        expr.name+"_linearized", expr_diff)
                    
                    # Solve "exact" using exp
                    self.add_indexed_expression(\
                        result_name, (i,), expr.state.sym+expr.sym/linearized*\
                        (sp.exp(linearized*dt)-1.0), offset_str)
                    
                else:
                    # Explicit Euler step
                    self.add_indexed_expression(result_name, (i,), \
                                    expr.state.sym+dt*expr.sym, offset_str)
                    
            elif solver_expr_added and recount:
                expr._recount()

        # Call recreate body with the jacobian expressions as the result
        # expressions
        results = {result_name:self.indexed_objects(result_name)}
        results, body_expressions = self._body_from_results(**results)
        self.body_expressions = self._recreate_body(\
            body_expressions, **results)

class SimplifiedImplicitEuler(CodeComponent):
    """
    An ODEComponent which compute one step of a simplified Implicit Euler
    algorithm
    """
    def __init__(self, ode, function_name="forward_simplified_implicit_euler", \
                 params=None):
        """
        Create a JacobianComponent

        Arguments
        ---------
        ode : ODE
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
        descr = "Compute a forward step using the simplified implicit Euler"\
                "algorithm to the {0} ODE".format(ode)
        super(SimplifiedImplicitEuler, self).__init__(\
            "SimplifiedImplicitEuler", ode, function_name, descr, params=params,
            additional_arguments=["dt"])

        # Recount the expressions if representation of states are "array" as
        # then the method is not full explcit
        recount = self._params.states.representation != "array"

        # Gather state expressions and states
        state_exprs = self.root.state_expressions
        states = self.root.full_states
        state_dict = dict((state.sym, ind) for ind, state in enumerate(states))
        all_exprs = sorted(self.root.intermediates + self.root.comments \
                           + state_exprs)

        result_name = self._params.states.array_name
        state_offset = self._params.states.add_offset
        self.shapes[result_name] = (len(states),)
        
        self.add_comment("Computing the explicit Euler algorithm")

        # Get time step and start creating the update algorithm
        if self._params.states.add_offset:
            offset_str = "{0}_offset".format(result_name)
        else:
            offset_str = ""
        
        solver_expr_added = False
        total_num_state_expressions = sum(isinstance(expr, StateExpression) \
                                          for expr in all_exprs)
        dt = self.root._dt.sym
        num_state_expressions = 0
        for expr in all_exprs:

            # Check if we are finished
            if num_state_expressions == total_num_state_expressions:
                break
            
            if isinstance(expr, StateExpression):
                num_state_expressions += 1
                i = state_dict[expr.state.sym]
                solver_expr_added = True

                if recount:
                    expr._recount()

                # Diagonal jacobian value
                diag_jac_expr = expr.expr.diff(expr.state.sym)

                if not diag_jac_expr.is_zero:
                    diag_jac = self.add_intermediate(\
                        expr.name+"_diag_jac", diag_jac_expr)
                else:
                    diag_jac = 0.

                # Add simplified single Implicit Euler step
                self.add_indexed_expression(result_name, (i,), \
                    expr.state.sym+dt*expr.sym/(1-dt*diag_jac), offset_str)
                    
            elif solver_expr_added and recount:
                expr._recount()

        # Call recreate body with the jacobian expressions as the result
        # expressions
        results = {result_name:self.indexed_objects(result_name)}
        results, body_expressions = self._body_from_results(**results)
        self.body_expressions = self._recreate_body(\
            body_expressions, **results)

