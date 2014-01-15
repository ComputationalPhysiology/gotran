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

__all__ = ["JacobianComponent", "JacobianActionComponent", \
           "FactorizedJacobianComponent", \
           "ForwardBackwardSubstitutionComponent",
           "LinearizedDerivativeComponent",
           "CommonSubExpressionODE", "componentwise_derivative", \
           "linearized_derivatives", "jacobian_expressions", \
           "jacobian_action_expressions", "factorized_jacobian_expressions",
           "forward_backward_subst_expressions",\
           "diagonal_jacobian_expressions", "rhs_expressions",\
           "diagonal_jacobian_action_expressions", \
           "monitored_expressions"]

# System imports
import sys

# ModelParameters imports
from modelparameters.sympytools import sp

# Local imports
from gotran.common import error, info, debug, check_arg, check_kwarg, \
     scalars, Timer, warning, tuplewrap, parameters
from gotran.model.utils import ode_primitives
from gotran.model.odeobjects2 import State, Parameter, IndexedObject, Comment
from gotran.model.expressions2 import *
from gotran.model.ode2 import ODE
from gotran.codegeneration.codecomponent import CodeComponent

#FIXME: Remove our own cse, or move to this module?
from gotran.codegeneration.sympy_cse import cse

def rhs_expressions(ode, function_name="rhs", result_name="dy"):
    """
    Return a code component with body expressions for the right hand side

    Arguments
    ---------
    ode : ODE
        The finalized ODE
    function_name : str
        The name of the function which should be generated
    result_name : str
        The name of the variable storing the rhs result
    """

    check_arg(ode, ODE)
    if not ode.is_finalized:
        error("Cannot compute right hand side expressions if the ODE is "\
              "not finalized")
    
    descr = "Compute the right hand side of the {0} ODE".format(ode)
    
    return CodeComponent("RHSComponent", ode, function_name, descr,\
                         **{result_name:ode.state_expressions})

def monitored_expressions(ode, monitored, function_name="monitored_expressions",
                          result_name="monitored"):
    """
    Return a code component with body expressions to calculate monitored expressions

    Arguments
    ---------
    ode : ODE
        The finalized ODE for which the monitored expression should be computed
    monitored : tuple, list
        A tuple/list of strings containing the name of the monitored expressions
    function_name : str
        The name of the function which should be generated
    result_name : str
        The name of the variable storing the rhs result
    """

    check_arg(ode, ODE)
    if not ode.is_finalized:
        error("Cannot compute right hand side expressions if the ODE is "\
              "not finalized")

    check_arg(monitored, (tuple, list), itemtypes=str)
    monitored_exprs = []
    for expr_str in monitored:
        obj = ode.present_ode_objects.get(expr_str)
        if not isinstance(obj, Expression):
            error("{0} is not an expression in the {1} ODE".format(expr_str, ode))
        
        monitored_exprs.append(obj)

    descr = "Computes monitored expressions of the {0} ODE".format(ode)
    return CodeComponent("MonitoredExpressions", ode, function_name, descr, \
                         **{result_name:monitored_exprs})
    
def componentwise_derivative(ode, index):
    """
    Return an ODEComponent holding the expressions for the ith
    state derivative

    Arguments
    ---------
    ode : ODE
        The finalized ODE for which the ith derivative should be computed
    index : int
        The index
    """
    check_arg(ode, ODE)
    if not ode.is_finalized:
        error("Cannot compute component wise derivatives if ODE is "\
              "not finalized")

    check_arg(index, int, ge=0, le=ode.num_full_states)

    # Get state expression
    expr = ode.state_expressions[index]
    state = expr.state
    if not isinstance(expr, StateDerivative):
        error("The ith index is not a StateDerivative: {0}".format(expr))
        
    return CodeComponent("d{0}_dt_component".format(\
        state), ode, "", "", dy=[expr])

def linearized_derivatives(ode, function_name="linear_derivatives", \
                           result_name="linearized"):
    """
    Return an ODEComponent holding the linearized derivative expressions

    Arguments
    ---------
    ode : ODE
        The ODE for which derivatives should be linearized
    function_name : str
        The name of the function which should be generated
    result_name : str
        The name of the variable storing the linearized derivatives
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return LinearizedDerivativeComponent(ode, function_name, result_name)

def jacobian_expressions(ode, function_name="compute_jacobian", result_name="jac"):
    """
    Return an ODEComponent holding expressions for the jacobian

    Arguments
    ---------
    ode : ODE
        The ODE for which the jacobian expressions should be computed
    function_name : str
        The name of the function which should be generated
    result_name : str
        The name of the variable storing the jacobian result
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return JacobianComponent(ode, function_name=function_name, \
                             result_name=result_name)

def jacobian_action_expressions(jacobian, with_body=True, \
                                function_name="compute_jacobian_action",\
                                result_name="jac_action"):
    """
    Return an ODEComponent holding expressions for the jacobian action

    Arguments
    ---------
    jacobian : JacobianComponent
        The ODEComponent holding expressions for the jacobian
    with_body : bool
        If true, the body for computing the jacobian will be included 
    function_name : str
        The name of the function which should be generated
    result_name : str
        The name of the variable storing the jacobian diagonal result
    """
    
    check_arg(jacobian, JacobianComponent)
    return JacobianActionComponent(jacobian, with_body, function_name, \
                                   result_name)

def diagonal_jacobian_expressions(jacobian, function_name="compute_diagonal_jacobian", \
                                  result_name="diag_jac"):
    """
    Return an ODEComponent holding expressions for the diagonal jacobian

    Arguments
    ---------
    jacobian : JacobianComponent
        The Jacobian of the ODE
    function_name : str
        The name of the function which should be generated
    result_name : str
        The name of the variable storing the jacobian diagonal result
    """
    return DiagonalJacobianComponent(jacobian, function_name, result_name)

def diagonal_jacobian_action_expressions(diagonal_jacobian, with_body=True, \
                                         function_name="compute_diagonal_jacobian_action",\
                                         result_name="diag_jac_action"):
    """
    Return an ODEComponent holding expressions for the diagonal jacobian action

    Arguments
    ---------
    diagonal_jacobian : DiagonalJacobianComponent
        The ODEComponent holding expressions for the diagonal jacobian
    with_body : bool
        If true, the body for computing the jacobian will be included 
    function_name : str
        The name of the function which should be generated
    result_name : str
        The name of the variable storing the jacobian diagonal result
    """
    
    check_arg(diagonal_jacobian, DiagonalJacobianComponent)
    return DiagonalJacobianActionComponent(diagonal_jacobian, with_body, function_name, \
                                           result_name)

def factorized_jacobian_expressions(jacobian, jacobian_name="jac"):
    """
    Return an ODEComponent holding expressions for the factorized jacobian

    Arguments
    ---------
    jacobian : JacobianComponent
        The ODEComponent holding expressions for the jacobian
    jacobian_name : str (optional)
        The basename of the jacobian name used in the jacobian component
    """
    check_arg(jacobian, JacobianComponent)
    return FactorizedJacobianComponent(jacobian, jacobian_name)

def forward_backward_subst_expressions(factorized):
    """
    Return an ODEComponent holding expressions for the forward backward
    substitions for a factorized jacobian

    Arguments
    ---------
    factoriced : FactorizedJacobianComponent
        The ODEComponent holding expressions for the factorized jacobian
    """
    check_arg(factoriced, FactorizedJacobianComponent)
    return ForwardBackwardSubstitutionComponent(jacobian)

class JacobianComponent(CodeComponent):
    """
    An ODEComponent which keeps all expressions for the Jacobian of the rhs
    """
    def __init__(self, ode, function_name="compute_jacobian", result_name="jac"):
        """
        Create a JacobianComponent

        Arguments
        ---------
        ode : ODE
            The parent component of this ODEComponent
        function_name : str
            The name of the function which should be generated
        result_name : str
            The name of the variable storing the jacobian result
        """
        check_arg(ode, ODE)

        # Call base class using empty result_expressions
        descr = "Compute the jacobian of the right hand side of the "\
                "{0} ODE".format(ode)
        super(JacobianComponent, self).__init__("Jacobian", ode, function_name, \
                                                descr)

        check_arg(result_name, str)

        timer = Timer("Computing jacobian")
        
        # Gather state expressions and states
        state_exprs = self.root.state_expressions
        states = self.root.full_states

        # Create Jacobian matrix
        N = len(states)
        self.jacobian = sp.Matrix(N, N, lambda i, j : 0.0)
        
        self.num_nonzero = 0

        self.add_comment("Computing the sparse jacobian of {0}".format(ode.name))
        self.shapes[result_name] = (N,N)
        
        state_dict = dict((state.sym, ind) for ind, state in enumerate(states))
        time_sym = states[0].time.sym
        
        might_take_time = N >= 10

        if might_take_time:
            info("Calculating Jacobian of {0}. Might take some time...".format(\
                ode.name))
            sys.stdout.flush()

        for i, expr in enumerate(state_exprs):
            states_syms = [sym for sym in ode_primitives(expr.expr, time_sym) \
                           if sym in state_dict]
            
            for sym in states_syms:
                j = state_dict[sym]
                time_diff = Timer("Differentiate state_expressions")
                jac_ij = expr.expr.diff(sym)
                del time_diff
                self.num_nonzero += 1
                jac_ij = self.add_indexed_expression(result_name, \
                                                     (i, j), jac_ij)
                self.jacobian[i, j] = jac_ij

        if might_take_time:
            info(" done")

        # Call recreate body with the jacobian expressions as the result
        # expressions
        results = {result_name:self.indexed_objects(result_name)}
        results, body_expressions = self._body_from_results(**results)
        self.body_expressions = self._recreate_body(\
            body_expressions, **results)

class DiagonalJacobianComponent(CodeComponent):
    """
    An ODEComponent which keeps all expressions for the Jacobian of the rhs
    """
    def __init__(self, jacobian, function_name="compute_diagonal_jacobian", \
                 result_name="diag_jac"):
        """
        Create a DiagonalJacobianComponent

        Arguments
        ---------
        jacobian : JacobianComponent
            The Jacobian of the ODE
        function_name : str
            The name of the function which should be generated
        result_name : str (optional)
            The basename of the indexed result expression
        """
        check_arg(jacobian, JacobianComponent)
        
        descr = "Compute the diagonal jacobian of the right hand side of the "\
                "{0} ODE".format(jacobian.root)
        super(DiagonalJacobianComponent, self).__init__(\
            "DiagonalJacobian", jacobian.root, function_name, descr)

        what = "Computing diagonal jacobian"
        timer = Timer(what)

        self.add_comment(what)

        N = jacobian.jacobian.shape[0]
        self.shapes[result_name] = (N,)
        jacobian_name = jacobian.results[0]

        # Create IndexExpressions of the diagonal Jacobian
        for expr in jacobian.indexed_objects(jacobian_name):
            if expr.indices[0]==expr.indices[1]:
                self.add_indexed_expression(result_name, expr.indices[0], \
                                            expr.expr)

        self.diagonal_jacobian = sp.Matrix(N, N, lambda i, j : 0.0)

        for i in range(N):
            self.diagonal_jacobian[i,i] = jacobian.jacobian[i,i]

        # Call recreate body with the jacobian diagonal expressions as the 
        # result expressions
        results = {result_name:self.indexed_objects(result_name)}
        results, body_expressions = self._body_from_results(**results)
        self.body_expressions = self._recreate_body(\
            body_expressions, **results)

class JacobianActionComponent(CodeComponent):
    """
    Jacobian action component which returns the expressions for Jac*x
    """
    def __init__(self, jacobian, with_body=True, \
                 function_name="compute_jacobian_action", \
                 result_name="jac_action"):
        """
        Create a JacobianActionComponent

        Arguments
        ---------
        jacobian : JacobianComponent
            The Jacobian of the ODE
        with_body : bool
            If true, the body for computing the jacobian will be included 
        function_name : str
            The name of the function which should be generated
        result_name : str
            The basename of the indexed result expression
        """
        timer = Timer("Computing jacobian action component")
        check_arg(jacobian, JacobianComponent)
        descr = "Compute the jacobian action of the right hand side of the "\
                "{0} ODE".format(jacobian.root)
        super(JacobianActionComponent, self).__init__(\
            "JacobianAction", jacobian.root, function_name, descr)

        x = self.root.full_state_vector
        jac = jacobian.jacobian
        jacobian_name = jacobian.results[0]

        # Create Jacobian action vector
        self.action_vector = sp.Matrix(len(x), 1,lambda i,j:0)

        self.add_comment("Computing the action of the jacobian")

        self.shapes[result_name] = (len(x),)
        self.shapes[jacobian_name] = jacobian.shapes[jacobian_name]
        for i, expr in enumerate(jac*x):
            self.action_vector[i] = self.add_indexed_expression(result_name,\
                                                                 i, expr)

        # Call recreate body with the jacobian action expressions as the 
        # result expressions
        results = {result_name:self.indexed_objects(result_name)}
        if with_body:
            results, body_expressions = self._body_from_results(**results)
        else:
            body_expressions = results[result_name]
        
        self.body_expressions = self._recreate_body(\
            body_expressions, **results)

class DiagonalJacobianActionComponent(CodeComponent):
    """
    Jacobian action component which returns the expressions for Jac*x
    """
    def __init__(self, diagonal_jacobian, with_body=True, \
                 function_name="compute_diagonal_jacobian_action", \
                 result_name="diag_jac_action"):
        """
        Create a DiagonalJacobianActionComponent

        Arguments
        ---------
        jacobian : JacobianComponent
            The Jacobian of the ODE
        with_body : bool
            If true, the body for computing the jacobian will be included 
        function_name : str
            The name of the function which should be generated
        result_name : str
            The basename of the indexed result expression
        """
        timer = Timer("Computing jacobian action component")
        check_arg(diagonal_jacobian, DiagonalJacobianComponent)
        descr = "Compute the diagonal jacobian action of the right hand side "\
                "of the {0} ODE".format(diagonal_jacobian.root)
        super(DiagonalJacobianActionComponent, self).__init__(\
            "DiagonalJacobianAction", diagonal_jacobian.root, function_name, descr)

        x = self.root.full_state_vector
        jac = diagonal_jacobian.diagonal_jacobian

        self._action_vector = sp.Matrix(len(x), 1,lambda i,j:0)

        self.add_comment("Computing the action of the jacobian")

        # Create Jacobian matrix
        self.shapes[result_name] = (len(x),)
        for i, expr in enumerate(jac*x):
            self._action_vector[i] = self.add_indexed_expression(\
                result_name, i, expr)

        # Call recreate body with the jacobian action expressions as the 
        # result expressions
        results = {result_name:self.indexed_objects(result_name)}
        if with_body:
            results, body_expressions = self._body_from_results(**results)
        else:
            body_expressions = results[result_name]
        
        self.body_expressions = self._recreate_body(\
            body_expressions, **results)

class FactorizedJacobianComponent(CodeComponent):
    """
    Class to generate expressions for symbolicaly factorizing a jacobian
    """
    def __init__(self, jacobian, function_name="factorize_jacobian"):
        """
        Create a FactorizedJacobianComponent
        """
        
        timer = Timer("Computing factorization of jacobian")
        check_arg(jacobian, JacobianComponent)
        descr = "Symbolicly factorize the jacobian of the {0} ODE".format(\
            jacobian.root)
        super(FactorizedJacobianComponent, self).__init__(\
            "FactorizedJacobian", jacobian.root, function_name, descr)

        self.add_comment("Factorizing jacobian of {0}".format(jacobian.root.name))
        
        jacobian_name = jacobian.results[0]

        # Get copy of jacobian
        jac = jacobian.jacobian[:,:]
        p = []

        # Size of system
        n = jac.rows

        self.shapes[jacobian_name] = (n,n)
        def add_intermediate_if_changed(jac, jac_ij, i, j):
            # If item has changed 
            if jac_ij != jac[i,j]:
                jac[i,j] = self.add_indexed_expression(jacobian_name, (i, j), jac_ij)

        # Do the factorization
        for j in range(n):
            for i in range(j):
                
                # Get sympy expr of A_ij
                jac_ij = jac[i,j]

                # Build sympy expression
                for k in range(i):
                    jac_ij -= jac[i,k]*jac[k,j]

                add_intermediate_if_changed(jac, jac_ij, i, j)
                    
            pivot = -1

            for i in range(j, n):

                # Get sympy expr of A_ij
                jac_ij = jac[i,j]

                # Build sympy expression
                for k in range(j):
                    jac_ij -= jac[i,k]*jac[k,j]

                add_intermediate_if_changed(jac, jac_ij, i, j)

                # find the first non-zero pivot, includes any expression
                if pivot == -1 and jac[i,j]:
                    pivot = i
                
            if pivot < 0:
                # this result is based on iszerofunc's analysis of the
                # possible pivots, so even though the element may not be
                # strictly zero, the supplied iszerofunc's evaluation gave
                # True
                error("No nonzero pivot found; symbolic inversion failed.")

            if pivot != j: # row must be swapped
                jac.row_swap(pivot,j)
                p.append([pivot,j])
                print "Pivoting!!"

            # Scale with diagonal
            if not jac[j,j]:
                error("Diagonal element of the jacobian is zero. "\
                      "Inversion failed")
                
            scale = 1 / jac[j,j]
            for i in range(j+1, n):
                
                # Get sympy expr of A_ij
                jac_ij = jac[i,j]
                jac_ij *= scale
                add_intermediate_if_changed(jac, jac_ij, i, j)

        # Store factorized jacobian
        self.factorized_jacobian = jac
        self.num_nonzero = sum(not jac[i,j].is_zero for i in range(n) \
                                for j in range(n))

        # No need to call recreate body expressions
        self.body_expressions = self.ode_objects
     
class ForwardBackwardSubstitutionComponent(CodeComponent):
    """
    Class to generate a forward backward substiution algorithm for
    symbolically factorized jacobian
    """
    def __init__(self, factorized, jacobian_name="jac", residual_name="dx"):
        """
        Create a JacobianForwardBackwardSubstComponent
        """
        check_arg(factorized, FactorizedJacobianComponent)
        super(ForwardBackwardSubstitutionComponent, self).__init__(\
            "ForwardBackwardSubst", factorized.root)
        check_arg(parent, ODE)

class LinearizedDerivativeComponent(CodeComponent):
    """
    A component for all linear and linearized derivatives
    """
    def __init__(self, ode, function_name="linear_derivatives", \
                 result_name="linearized"):

        descr = "Computes the linearized derivatives for all linear derivatives"
        super(LinearizedDerivativeComponent, self).__init__(\
            "LinearizedDerivatives", ode, function_name, descr)
        
        check_arg(ode, ODE)
        assert ode.is_finalized
        self.linear_derivative_indices = [0]*self.root.num_full_states
        self.shapes[result_name] = (self.root.num_full_states,)
        for ind, expr in enumerate(self.root.state_expressions):
            if not isinstance(expr, StateDerivative):
                error("Cannot generate a linearized derivative of an "\
                      "algebraic expression.")
            expr_diff = expr.expr.diff(expr.state.sym)

            if expr_diff and expr.state.sym not in expr_diff:

                self.linear_derivative_indices[ind] = 1
                self.add_indexed_expression("linearized", ind, expr_diff)

        # Call recreate body with the jacobian action expressions as the 
        # result expressions
        results = {result_name:self.indexed_objects(result_name)}
        results, body_expressions = self._body_from_results(**results)
        self.body_expressions = self._recreate_body(body_expressions, \
                                                    **results)

class CommonSubExpressionODE(ODE):
    """
    Class which flattens the component structue of an ODE to just one.
    It uses common sub expressions as intermediates to reduce complexity
    of the derivative expressions.
    """
    def __init__(self, ode):
        check_arg(ode, ODE)
        assert ode.is_finalized
        
        timer = Timer("Extract common sub expressions")

        newname = ode.name+"_CSE"
        
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
        expanded_state_exprs = [ode.expanded_expressions[obj.name] \
                                for obj in org_state_expressions]

        # Call sympy common sub expression reduction
        cse_exprs, cse_state_exprs = cse(expanded_state_exprs,\
                                         symbols=sp.numbered_symbols("cse_"),\
                                         optimizations=[])
        cse_cnt = 0
        cse_subs = {}

        # Register the common sub expressions as Intermediates
        for sub, expr in cse_exprs:
        
            # If the expression is just one of the atoms of the ODE we skip
            # the cse expressions but add a subs for the atom
            if expr in atoms:
                cse_subs[sub] = expr
            else:
                cse_subs[sub] =  self.add_intermediate("cse_{0}".format(\
                    cse_cnt), expr.xreplace(cse_subs))
                cse_cnt += 1

        # Register the state expressions
        for org_state_expr, state_expr in \
                zip(org_state_expressions, cse_state_exprs):

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


