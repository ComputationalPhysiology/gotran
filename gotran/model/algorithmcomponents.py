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

__all__ = ["JacobianComponent", "JacobianActionComponent", \
           "JacobianFactorizationComponent", \
           "JacobianForwardBackwardSubstComponent",
           "DependentExpressionComponent", "LinearizedDerivativeComponent"]

# System imports

# ModelParameters imports
from modelparameters.sympytools import sp

# Local imports
from gotran.common import error, debug, check_arg, check_kwarg, scalars
from odeobjects2 import State, Parameter
from expressions2 import Expression, StateDerivative
from odecomponents2 import ODEBaseComponent, ODE

class JacobianComponent(ODEBaseComponent):
    """
    An ODEComponent which keeps all expressions for the Jacobian of the rhs
    """
    def __init__(self, parent):
        """
        Create a JacobianComponent

        Arguments
        ---------
        name : str
            The name of the component. This str serves as the unique
            identifier of the Component.
        parent : ODEBaseComponent
            The parent component of this ODEComponent
        """
        super(JacobianComponent, self).__init__("Jacobian", parent)
        check_arg(parent, ODE)

        # Gather state expressions and states
        state_exprs = self.root.state_expressions
        states = self.root.full_states

        # Create Jacobian matrix
        N = len(states)
        self._jacobian = sp.Matrix(N, N, lambda i, j : 0.0)
        
        self._num_nonzero = 0
        
        for i, expr in enumerate(state_exprs):
            for j, state in enumerate(states):

                jac_ij = expr.expr.diff(state.sym)

                # Only collect non zero contributions
                if jac_ij:
                    self._num_nonzero += 1
                    jac_ij = self.add_intermediate("jac_{0}_{1}".format(i,j), jac_ij)
                    self._jacobian[i, j] = jac_ij

        # Attributes for jacobian action and jacobian solution components
        self._action_component = None
        self._factorization_component = None

    @property
    def jacobian(self):
        """
        Return the jacobian matrix
        """
        return self._jacobian

    @property
    def num_nonzero(self):
        """
        Return the num non zeros of the Jacobian
        """
        return self._num_nonzero


    def action_component(self):
        """
        Return a jacobian action component
        """

        if self._action_component is None:
            self._action_component = JacobianActionComponent(self.root)

        return self._action_component

    def factorization_component(self):
        """
        Return a jacobian factorization component
        """

        if self._factorization_component is None:
            self._factorization_component = \
                        JacobianFactorizationComponent(self.root)

        return self._factorization_component

    def forward_backward_subst_component(self):
        """
        Return a jacobian forward_backward_subst component
        """

        if self._forward_backward_subst_component is None:
            self._forward_backward_subst_component = \
                        JacobianForwardBackwardSubstComponent(self.root)

        return self._forward_backward_subst_component
        

class JacobianActionComponent(ODEBaseComponent):
    """
    Jacobian action component which returns the expressions for Jac*x
    """
    def __init__(self, parent):
        """
        Create a JacobianActionComponent
        """
        super(JacobianActionComponent, self).__init__("JacobianAction", parent)
        check_arg(parent, ODE)

        x = self.root.full_state_vector
        jac = self.root.jacobian_component().jacobian

        self._action_vector = sp.Matrix(len(x), 1,lambda i,j:0)

        # Create Jacobian matrix
        for i, expr in enumerate(jac*x):
            self._action_vector[i] = self.add_intermediate(\
                "jac_action_{0}".format(i), expr)

    @property
    def action_vector(self):
        """
        Return the vector of all action symbols
        """
        return self._action_vector

class JacobianFactorizationComponent(ODEBaseComponent):
    """
    Class to generate expressions for symbolicaly factorize a jacobian
    """
    def __init__(self, parent):
        """
        Create a JacobianSymbolicFactorizationComponent
        """
        
        super(JacobianFactorizationComponent, self).__init__(\
            "JacobianFactorization", parent)

        # Get copy of jacobian
        jac = self.root.jacobian_component().jacobian[:,:]
        p = []

        # Size of system
        n = jac.rows

        def add_intermediate_if_changed(jac, jac_ij, i, j):
            # If item has changed 
            if jac_ij != jac[i,j]:
                jac[i,j] = self.add_intermediate(\
                    "jac_{0}_{1}".format(i,j), jac_ij)

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
        self._factorized_jacobian = jac

    @property
    def factorized_jacobian(self):
        return self._factorized_jacobian
        
class JacobianForwardBackwardSubstComponent(ODEBaseComponent):
    """
    Class to generate a forward backward substiution algorithm for
    symbolically factorized jacobian
    """
    def __init__(self, parent):
        """
        Create a JacobianForwardBackwardSubstComponent
        """
        
        super(JacobianForwardBackwardSubstComponent, self).__init__(\
            "JacobianForwardBackwardSubst", parent)
        check_arg(parent, ODE)

class DependentExpressionComponent(ODEBaseComponent):
    """
    Component which takes a set of expressions and extracts dependent
    expressions from the ODE
    """
    def __init__(self, name, parent, *args):
        """
        Create a DependentExpressionComponent

        Arguments
        ---------
        args : tuple of Expressions
        """
        super(DependentExpressionComponent, self).__init__(\
            name, parent)
        check_arg(parent, ODE)
        
        self.init_deps(*args)

    def init_deps(self, *args):
        """
        Init the dependencies
        """
        ode_expr_deps = self.root.expression_dependencies
        
        # Check passed args
        exprs = set()
        not_checked = set()
        used_states = set()
        used_parameters = set()
        used_field_parameters = set()

        exprs_not_in_body = []

        for i, arg in enumerate(args):
            check_arg(arg, Expression, i+1, DependentExpressionComponent.init_deps)

            # If arg is part of body
            if self.root._arg_indices.get(arg):
                exprs.add(arg)
            else:
                exprs_not_in_body.append(arg)
            
            # Collect dependencies
            for obj in ode_expr_deps[arg]:
                if isinstance(obj, Expression):
                    not_checked.add(obj)
                elif isinstance(obj, State):
                    used_states.add(obj)
                elif isinstance(obj, Parameter):
                    if obj.is_field:
                        used_field_parameters.add(obj)
                    else:
                        used_parameters.add(obj)

        # Collect all dependencies
        while not_checked:

            dep_expr = not_checked.pop()
            exprs.add(dep_expr)
            for obj in ode_expr_deps[dep_expr]:
                if isinstance(obj, Expression):
                    if obj not in exprs:
                        not_checked.add(obj)
                elif isinstance(obj, State):
                    used_states.add(obj)
                elif isinstance(obj, Parameter):
                    if obj.is_field:
                        used_field_parameters.add(obj)
                    else:
                        used_parameters.add(obj)

        # Sort used state, parameters and expr
        arg_cmp = self.root.arg_cmp
        self._used_states = sorted(used_states, arg_cmp)
        self._used_parameters = sorted(used_parameters, arg_cmp)
        self._used_field_parameters = sorted(used_field_parameters, arg_cmp)
        self._body_expressions = sorted(exprs, arg_cmp)
        self._body_expressions.extend(exprs_not_in_body)
        
    @property
    def used_states(self):
        return self._used_states

    @property
    def used_parameters(self):
        return self._used_parameters

    @property
    def used_field_parameters(self):
        return self._used_field_parameters

    @property
    def body_expressions(self):
        return self._body_expressions

class LinearizedDerivativeComponent(DependentExpressionComponent):
    """
    A component for all linear and linearized derivatives
    """
    def __init__(self, parent):

        super(LinearizedDerivativeComponent, self).__init__(\
            "LinearizedDerivatives", parent)
        
        check_arg(parent, ODE)
        assert parent.is_finalized
        self._linear_derivative_indices = [0]*self.root.num_full_states
        for ind, expr in enumerate(self.root.state_expressions):
            if not isinstance(expr, StateDerivative):
                error("Cannot generate a linearized derivative of an "\
                      "algebraic expression.")
            expr_diff = expr.expr.diff(expr.state.sym)

            if expr_diff and expr.state.sym not in expr_diff:

                self._linear_derivative_indices[ind] = 1
                self.add_intermediate("linearized_d{0}_dt".format(expr.state), expr_diff)

        self.init_deps(*self.intermediates)

    def linear_derivative_indices(self):
        return self._linear_derivative_indices
        
