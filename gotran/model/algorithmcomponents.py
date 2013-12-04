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
           "FactorizedJacobianComponent", \
           "ForwardBackwardSubstitutionComponent",
           "DependentExpressionComponent", "LinearizedDerivativeComponent",
           "CommonSubExpressionODE", "componentwise_derivative",
           "linearized_derivatives", "jacobian_expressions", \
           "jacobian_action_expressions", "factorized_jacobian_expressions",
           "forward_backward_subst_expressions",
           "diagonal_jacobian_expressions", "IndexedExpressionComponent"]

# System imports
from collections import OrderedDict

# ModelParameters imports
from modelparameters.sympytools import sp

# Local imports
from gotran.common import error, debug, check_arg, check_kwarg, scalars, Timer
from odeobjects2 import State, Parameter, IndexedObject
from expressions2 import Expression, StateDerivative, IndexedExpression
from odecomponents2 import ODEBaseComponent, ODE

#FIXME: Remove our own cse, or move to this module?
from gotran.codegeneration.sympy_cse import cse

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
    if not ode.is_finalized:
        error("Cannot compute component wise derivatives if ODE is "\
              "not finalized")

    check_arg(index, int, ge=0, le=ode.num_full_states)

    # Get state expression
    expr = ode.state_expressions[index]
    state = expr.state
    if not isinstance(expr, StateDerivative):
        error("The ith index is not a StateDerivative: {0}".format(expr))
        
    return DependentExpressionComponent("d{0}_dt_component".format(\
        state), ode, expr)

def linearized_derivatives(ode):
    """
    Return an ODEComponent holding the linearized derivative expressions

    Arguments
    ---------
    ode : ODE
        The ODE for which derivatives should be linearized
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return LinearizedDerivativeComponent(ode)

def jacobian_expressions(ode):
    """
    Return an ODEComponent holding expressions for the jacobian

    Arguments
    ---------
    ode : ODE
        The ODE for which the jacobian expressions should be computed
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return JacobianComponent(ode)

def diagonal_jacobian_expressions(jacobian):
    """
    Return an ODEComponent holding expressions for the diagonal jacobian

    Arguments
    ---------
    jacobian : JacobianComponent
        The Jacobian of the ODE
    """

    return DiagonalJacobianComponent(jacobian)

def jacobian_action_expressions(jacobian):
    """
    Return an ODEComponent holding expressions for the jacobian action

    Arguments
    ---------
    jacobian : JacobianComponent
        The ODEComponent holding expressions for the jacobian
    """
    
    check_arg(jacobian, JacobianComponent)
    return JacobianActionComponent(jacobian)

def factorized_jacobian_expressions(jacobian):
    """
    Return an ODEComponent holding expressions for the factorized jacobian

    Arguments
    ---------
    jacobian : JacobianComponent
        The ODEComponent holding expressions for the jacobian
    """
    check_arg(jacobian, JacobianComponent)
    return FactorizedJacobianComponent(jacobian)

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

class IndexedExpressionComponent(ODEBaseComponent):
    """
    An ODEComponent which allows adding indexed expressions
    """
    def __init__(self, name, parent):
        """
        Create an IndexedExpressionComponent

        Arguments
        ---------
        name : str
            The name of the component. This str serves as the unique
            identifier of the Component.
        parent : ODEBaseComponent
            The parent component of this ODEComponent
        """
        super(IndexedExpressionComponent, self).__init__(name, parent)

    def add_indexed_expression(self, basename, indices, expr):
        """
        Add an indexed expression using a basename and the indices

        Arguments
        ---------
        basename : str
            The basename of the indexed expression
        indices : int, tuple of int
            The fixed indices identifying the expression
        expr : sympy.Basic, scalar
            The expression
        """
        # Create an IndexedExpression in the present component
        timer = Timer("Add indexed expression")

        # Create indexed expression
        expr = IndexedExpression(basename, indices, expr)

        self._register_component_object(expr)

        return expr.sym

    def add_indexed_object(self, basename, indices):
        """
        Add an indexed object using a basename and the indices

        Arguments
        ---------
        basename : str
            The basename of the indexed expression
        indices : int, tuple of int
            The fixed indices identifying the expression
        """
        timer = Timer("Add indexed object")

        # Create state
        obj = IndexedObject(name, init, self.time)

        self._register_component_object(obj)

        # Return the sympy version of the state
        return obj.sym

    def indexed_expressions(self, basename):
        """
        Return a list of all indexed expressions with the given basename
        """
        return [obj for obj in self.ode_objects if isinstance(\
            obj, IndexedExpression) and obj.basename == basename]
        
    def indexed_objects(self, basename):
        """
        Return a list of all indexed objects with the given basename
        """
        return [obj for obj in self.ode_objects if isinstance(\
            obj, IndexedObject) and obj.basename == basename]
        
class DependentExpressionComponent(IndexedExpressionComponent):
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
        self._body_expressions = []
        self.init_deps(*args)

    def init_deps(self, *args):
        """
        Init the dependencies
        """

        if not args:
            return

        timer = Timer("Compute dependencies of a dependent expression component")
        
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
        self._used_states = sorted(used_states)
        self._used_parameters = sorted(used_parameters)
        self._used_field_parameters = sorted(used_field_parameters)
        self._body_expressions = sorted(list(exprs)+exprs_not_in_body)
        
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

class JacobianComponent(DependentExpressionComponent):
    """
    An ODEComponent which keeps all expressions for the Jacobian of the rhs
    """
    def __init__(self, parent):
        """
        Create a JacobianComponent

        Arguments
        ---------
        parent : ODE
            The parent component of this ODEComponent
        """
        super(JacobianComponent, self).__init__("Jacobian", parent)
        check_arg(parent, ODE)

        timer = Timer("Computing jacobian")
        
        # Gather state expressions and states
        state_exprs = self.root.state_expressions
        states = self.root.full_states

        # Create Jacobian matrix
        N = len(states)
        self._jacobian = sp.Matrix(N, N, lambda i, j : 0.0)
        
        self._num_nonzero = 0

        self.add_comment("Computing the sparse jacobian of {0}".format(parent.name))
        for i, expr in enumerate(state_exprs):
            for j, state in enumerate(states):

                jac_ij = expr.expr.diff(state.sym)

                # Only collect non zero contributions
                if jac_ij:
                    self._num_nonzero += 1
                    jac_ij = self.add_indexed_expression("jac", (i, j), jac_ij)
                    self._jacobian[i, j] = jac_ij

        self.init_deps(*self.indexed_expressions("jac"))

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

class DiagonalJacobianComponent(DependentExpressionComponent):
    """
    An ODEComponent which keeps all expressions for the Jacobian of the rhs
    """
    def __init__(self, jacobian):
        """
        Create a DiagonalJacobianComponent

        Arguments
        ---------
        jacobian : JacobianComponent
            The Jacobian of the ODE
        """
        check_arg(jacobian, JacobianComponent)
        super(DiagonalJacobianComponent, self).__init__(\
            "DiagonalJacobian", jacobian.root)

        timer = Timer("Computing diagonal jacobian")
        
        # Create Jacobian matrix
        diagonal_expressions = [expr for expr in \
                                jacobian.indexed_expressions("jac") if \
                                expr.indices[0]==expr.indices[1]]
        N = len(diagonal_expressions)

        self._diagonal_jacobian = sp.Matrix(N, N, lambda i, j : 0.0)

        for i in range(N):
            self._diagonal_jacobian = jacobian.jacobian[i,i]
            
        self.init_deps(*diagonal_expressions)

    @property
    def diagonal_jacobian(self):
        """
        Return the diagonal jacobian matrix
        """
        return self._diagonal_jacobian

class JacobianActionComponent(IndexedExpressionComponent):
    """
    Jacobian action component which returns the expressions for Jac*x
    """
    def __init__(self, jacobian):
        """
        Create a JacobianActionComponent
        """
        timer = Timer("Computing jacobian action component")
        check_arg(jacobian, JacobianComponent)
        super(JacobianActionComponent, self).__init__(\
            "JacobianAction", jacobian.root)

        x = self.root.full_state_vector
        jac = jacobian.jacobian

        self._action_vector = sp.Matrix(len(x), 1,lambda i,j:0)

        self.add_comment("Computing the action of the jacobian")

        # Create Jacobian matrix
        for i, expr in enumerate(jac*x):
            self._action_vector[i] = self.add_indexed_expression("jac_action",\
                                                                 i, expr)

        # Get body expressions from parent and extend with jacobian expressions
        self._body_expressions = jacobian.body_expressions[:]
        for obj in self.ode_objects:
            self._body_expressions.append(obj)

    @property
    def body_expressions(self):
        return self._body_expressions

    @property
    def action_vector(self):
        """
        Return the vector of all action symbols
        """
        return self._action_vector

class FactorizedJacobianComponent(IndexedExpressionComponent):
    """
    Class to generate expressions for symbolicaly factorize a jacobian
    """
    def __init__(self, jacobian):
        """
        Create a FactorizedJacobianComponent
        """
        
        timer = Timer("Computing factorization of jacobian")
        check_arg(jacobian, JacobianComponent)
        super(FactorizedJacobianComponent, self).__init__(\
            "FactorizedJacobian", jacobian.root)

        self.add_comment("Factorizing jacobian of {0}".format(jacobian.root.name))
        
        # Get copy of jacobian
        jac = jacobian.jacobian[:,:]
        p = []

        # Size of system
        n = jac.rows

        def add_intermediate_if_changed(jac, jac_ij, i, j):
            # If item has changed 
            if jac_ij != jac[i,j]:
                jac[i,j] = self.add_indexed_expression("jac", (i, j), jac_ij)

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
        self._num_nonzero = sum(not jac[i,j].is_zero for i in range(n) \
                                for j in range(n))

        self._body_expressions = []
        for obj in self.ode_objects:
            self._body_expressions.append(obj)
     

    @property
    def body_expressions(self):
        return self._body_expressions

    @property
    def num_nonzero(self):
        """
        Return the num non zeros of the Jacobian
        """
        return self._num_nonzero

    @property
    def factorized_jacobian(self):
        return self._factorized_jacobian
        
class ForwardBackwardSubstitutionComponent(IndexedExpressionComponent):
    """
    Class to generate a forward backward substiution algorithm for
    symbolically factorized jacobian
    """
    def __init__(self, factorized):
        """
        Create a JacobianForwardBackwardSubstComponent
        """
        check_arg(factorized, FactorizedJacobianComponent)
        super(ForwardBackwardSubstitutionComponent, self).__init__(\
            "ForwardBackwardSubst", factorized.root)
        check_arg(parent, ODE)

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
                self.add_indexed_expression("linearized", ind, expr_diff)

        self.init_deps(*self.indexed_expressions("linearized"))

    def linear_derivative_indices(self):
        return self._linear_derivative_indices
        

class CommonSubExpressionODE(ODE):
    """
    Class which flattens the component structue of an ODE to just one.
    It uses common sub expressions as intermediates to reduce complexity
    of the derivative expressions.
    """
    def __init__(self, ode):
        check_arg(ode, ODE)
        assert ode.is_finalized
        
        newname = ode.name+"_CSE"
        
        # Call super class 
        super(CommonSubExpressionODE, self).__init__(newname, ode.ns)

        # Add states and parameters
        atoms = []
        for state in ode.full_states:
            atoms.append(self.add_state(state.name, state.param))

        for param in ode.all_parameters:
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
