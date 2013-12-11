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
           "CommonSubExpressionODE", "ReuseBodyVariableComponent",
           "CodeComponent", \
           "componentwise_derivative", \
           "linearized_derivatives", "jacobian_expressions", \
           "jacobian_action_expressions", "factorized_jacobian_expressions",
           "forward_backward_subst_expressions",
           "diagonal_jacobian_expressions",
           "reuse_body_variables"]

# System imports
from collections import OrderedDict, deque, defaultdict
from sympy.core.function import AppliedUndef

# ModelParameters imports
from modelparameters.sympytools import sp
from modelparameters.codegeneration import sympycode

# Local imports
from gotran.common import error, debug, check_arg, check_kwarg, scalars, Timer, \
     warning, tuplewrap
from utils import ode_primitives
from odeobjects2 import State, Parameter, IndexedObject, Comment
from expressions2 import *
from odecomponents2 import ODEBaseComponent, ODE

#FIXME: Remove our own cse, or move to this module?
from gotran.codegeneration.sympy_cse import cse

def rhs_expressions(ode):
    """
    Return a right hand side code component 

    Arguments
    ---------
    ode : ODE
        The finalized ODE for which the ith derivative should be computed
    """

    check_arg(ode, ODE)
    if not ode.is_finalized:
        error("Cannot compute right hand side expressions if the ODE is "\
              "not finalized")
        
    return RHSComponent(ode)
    
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

def reuse_body_variables(component, *classes):
    """
    Function to reuse as much body variables as possible

    Arguments
    ---------
    component : ODEComonent
        An ODEComponent with the bode_expression attribute
    classes : tuple
        A tuple of expression classes to include in the body reuse
    """
    check_arg(classes, tuple, itemtypes=type)
    return ReuseBodyVariableComponent(component, *classes)

class CodeComponent(ODEBaseComponent):
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
        super(CodeComponent, self).__init__(name, parent)
        check_arg(parent, ODE)
        self.shapes = {}
        self._body_expressions = []

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
            The expression.
        """
        # Create an IndexedExpression in the present component
        timer = Timer("Add indexed expression")

        indices = tuplewrap(indices)

        # Check that provided indices fit with the registered shape
        if len(self.shapes[basename]) > len(indices):
            error("Shape missmatch between indices {0} and registered "\
                  "shape for {1}".format(indices, basename))

        for dim, (index, shape_ind) in enumerate(zip(indices, self.shapes[basename])):
            if index >= shape_ind:
                error("Indices must be smaller or equal to the shape. Missmatch "\
                      "in dim {0}: {1}>={2}".format(dim+1, index, shape_ind))

        # Create the indexed expression
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

        indices = tuplewrap(indices)

        # Create IndexedObject
        obj = IndexedObject(basename, indices)

        # Check that provided indices fit with the registered shape
        if len(self.shapes[basename]) > len(indices):
            error("Shape missmatch between indices {0} and registered "\
                  "shape for {1}".format(indices, basename))

        for dim, (index, shape_ind) in enumerate(zip(indices, self.shapes[basename])):
            if index >= shape_ind:
                error("Indices must be smaller or equal to the shape. Missmatch "\
                      "in dim {0}: {1}>={2}".format(dim+1, index, shape_ind))

        self._register_component_object(obj)

        # Return the sympy version of the state
        return obj.sym

    def indexed_expressions(self, *basenames):
        """
        Return a list of all indexed expressions with the given basename,
        if no base names give all indexed expressions are returned
        """
        if not basenames:
            basenames = self.shapes.keys()
        return [obj for obj in self.ode_objects if isinstance(\
            obj, IndexedExpression) and obj.basename in basenames]
        
    def indexed_objects(self, *basenames):
        """
        Return a list of all indexed objects with the given basename,
        if no base names give all indexed objects are returned
        """
        if not basenames:
            basenames = self.shapes.keys()
        return [obj for obj in self.ode_objects if isinstance(\
            obj, IndexedObject) and obj.basename in basenames]

    @property
    def body_expressions(self):
        """
        Return a list of all body expressions 
        """
        return self._body_expressions

class DependentExpressionComponent(CodeComponent):
    """
    Component which takes a set of expressions and extracts dependent
    expressions from the ODE
    """
    def __init__(self, name, parent, *result_expressions):
        """
        Create a DependentExpressionComponent

        Arguments
        ---------
        result_expressions : tuple of Expressions
            A tuple of expressions which will be the computed result of
            this Component
        """
        super(DependentExpressionComponent, self).__init__(\
            name, parent)
        check_arg(parent, ODE)
        self._body_expressions = []
        self.init_deps(*result_expressions)

    def init_deps(self, *result_expressions):
        """
        Init the dependencies
        """

        if not result_expressions:
            return

        timer = Timer("Compute dependencies for {0}".format(self.name))
        
        ode_expr_deps = self.root.expression_dependencies
        
        # Check passed expressions
        exprs = set(result_expressions)
        not_checked = set()
        used_states = set()
        used_parameters = set()

        exprs_not_in_body = []

        for expr in result_expressions:
            check_arg(expr, (Expression, Comment), \
                      context=DependentExpressionComponent.init_deps)
            
            if isinstance(expr, Comment):
                continue

            #if expr not in ode_expr_deps:
            #    error("The result expression {0} is not an expression of "\
            #          "the {1} ODE".format(expr, self.root))
            
            # Collect dependencies
            for obj in ode_expr_deps[expr]:
                if isinstance(obj, (Expression, Comment)):
                    not_checked.add(obj)
                elif isinstance(obj, State):
                    used_states.add(obj)
                elif isinstance(obj, Parameter):
                    used_parameters.add(obj)

        # Collect all dependencies
        while not_checked:

            dep_expr = not_checked.pop()
            exprs.add(dep_expr)
            for obj in ode_expr_deps[dep_expr]:
                if isinstance(obj, (Expression, Comment)):
                    if obj not in exprs:
                        not_checked.add(obj)
                elif isinstance(obj, State):
                    used_states.add(obj)
                elif isinstance(obj, Parameter):
                    used_parameters.add(obj)

        # Sort used state, parameters and expr
        self._used_states = sorted(used_states)
        self._used_parameters = sorted(used_parameters)
        self._body_expressions = sorted(list(exprs))
        self._result_expressions = list(result_expressions)
        
    @property
    def used_states(self):
        return self._used_states

    @property
    def used_parameters(self):
        return self._used_parameters

    @property
    def result_expressions(self):
        """
        Returns the expressions which are expected to be computed by
        the body expressions
        """
        return self._result_expressions

class RHSComponent(DependentExpressionComponent):
    """
    Dependent expression component for the RHS of an ODE
    """
    def __init__(self, parent, result_name="dy"):
        """
        Create an RHSComponent 

        Arguments
        ---------
        parent : ODE
            The parent component of this ODEComponent
        """
        check_arg(parent, ODE)
        super(RHSComponent, self).__init__("RHS", parent)

        check_arg(result_name, str)

        timer = Timer("Computing RHS Body expressions")

        # First init body expressions using the state expressions
        self.init_deps(self.root.state_expressions)

        # Second rename all collected state_expressions with indexed
        # expressions
        

    def init_bodyexpressions(self, result_name):
        """
        Build a list of all body expressions in the ODE. The result of the RHS
        is all StateExpressions. These will be replaced by IndexedExpressions.

        This is facilitated by a big hack. If we are going to change name of
        State expressions we need to:

        1) Add them as new Expressions
        2) Inherit expression_dependencies and object_used in from old
           state_expression
        3) Update the sympy expression in all expressions which use this
           expressions, and recursively to all expressions using these...
        """

        # Get a copy of all state expressions
        state_expressions = self.root.state_expressions
        
        # Create shape
        self.shape[result_name] = (len(state_expressions),)

        # Iterate over the state expressions in the order they were created
        # preserving interdependencies
        new_state_exprs = []
        state_expr_map = {}
        sym_replace = {}
        used_objects = defaultdict(list)
        for old_expr in sorted(state_expressions):

            # Get index based on the original ordering
            index = (state_expressions.index(state_expr),)

            # Create the IndexedExpression
            new_expr = IndexedExpression(result_name, index, old_expr.expr.\
                                         xreplace(sym_replace))

            # Store new expr in map
            state_expr_map[old_expr] = new_expr

            # Update sym replace dict
            sym_replace[old_expr.sym] = new_expr.sym
            
            # Copy counter from old expression so they sort properly
            new_expr._recount(old_expr._count)
            
            # Transfer and store dependencies
            new_deps = set(state_expr_map.get(dep_expr, dep_expr) \
                           for dep_expr in self.root.expression_dependencies[\
                            old_expr])
            self.root.expression_dependencies[new_expr] = new_deps

            # Store new expression
            new_state_exprs.append(new_expr)
            self.root.present_ode_objects[new_deps.name] = (new_expr, self)
            self.ode_objects.append(new_expr)

            # Collect expressions which need to be re-introduced as they depends 
            for using_expr in self.root.object_used_in[old_expr]:
                used_objects[used_objects].append(new_expression)
        
        body_expressions = []
        
        # Iterate over all components
        for comp_name in self.root.all_expr_components_ordered:

            comp = self.root.all_components[comp_name]

            # Iterate over all objects of the component
            for obj in comp.ode_objects:

                # Only add Expressions
                if isinstance(obj, (Expression, Comment)):
                    body_expressions.append(obj)
        
        self._body_expressions = body_expressions
        
        
class JacobianComponent(DependentExpressionComponent):
    """
    An ODEComponent which keeps all expressions for the Jacobian of the rhs
    """
    def __init__(self, parent, result_name="jac"):
        """
        Create a JacobianComponent

        Arguments
        ---------
        parent : ODE
            The parent component of this ODEComponent
        """
        check_arg(parent, ODE)
        super(JacobianComponent, self).__init__("Jacobian", parent)

        check_arg(result_name, str)

        timer = Timer("Computing jacobian")
        
        # Gather state expressions and states
        state_exprs = self.root.state_expressions
        states = self.root.full_states

        # Create Jacobian matrix
        N = len(states)
        self._jacobian = sp.Matrix(N, N, lambda i, j : 0.0)
        
        self._num_nonzero = 0

        self.add_comment("Computing the sparse jacobian of {0}".format(parent.name))
        self.shapes[result_name] = (N,N)
        
        state_dict = dict((state.sym, ind) for ind, state in enumerate(states))
        time_sym = states[0].time.sym
        
        for i, expr in enumerate(state_exprs):
            states_syms = [sym for sym in ode_primitives(expr.expr, time_sym) \
                           if sym in state_dict]
            
            for sym in states_syms:
                j = state_dict[sym]
                time_diff = Timer("Differentiate state_expressions")
                jac_ij = expr.expr.diff(sym)
                del time_diff
                self._num_nonzero += 1
                jac_ij = self.add_indexed_expression(result_name, \
                                                     (i, j), jac_ij)
                self._jacobian[i, j] = jac_ij
                
        # Call init deps with sorted list of objects
        self.init_deps(*self.indexed_expressions(result_name))
        
        self._result_name = result_name

    @property
    def result_name(self):
        return self._result_name

    @property
    def jacobian(self):
        """
        Return the jacobian matrix
        """
        return self._jacobian

    @property
    def argument_objects(self):
        """
        Returns a list of ODE objects which are treated as input arguments
        """
        return self.root.argument_objects

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
    def __init__(self, jacobian, result_name="diag_jac"):
        """
        Create a DiagonalJacobianComponent

        Arguments
        ---------
        jacobian : JacobianComponent
            The Jacobian of the ODE
        result_name : str
            The basename of the indexed result expression
        """
        check_arg(jacobian, JacobianComponent)
        super(DiagonalJacobianComponent, self).__init__(\
            "DiagonalJacobian", jacobian.root)

        what = "Computing diagonal jacobian"
        timer = Timer(what)

        self.add_comment(what)

        N = jacobian.jacobian.shape[0]
        self.shapes[result_name] = (N,)

        # Create IndexExpressions of the diagonal Jacobian
        for expr in jacobian.result_expressions:
            if expr.indices[0]==expr.indices[1]:
                self.add_indexed_expression(result_name, expr.indices[0], \
                                            expr.expr)

        self._diagonal_jacobian = sp.Matrix(N, N, lambda i, j : 0.0)

        for i in range(N):
            self._diagonal_jacobian[i,i] = jacobian.jacobian[i,i]

        # Add dependent body expressions
        self.init_deps(*self.indexed_expressions(result_name))

        self._result_name = result_name

    @property
    def diagonal_jacobian(self):
        """
        Return the diagonal jacobian matrix
        """
        return self._diagonal_jacobian

    def argument_objects(self):
        """
        Returns a list of ODE objects which are treated as input arguments
        """
        return self.root.argument_objects

    @property
    def result_name(self):
        return self._result_name

class JacobianActionComponent(CodeComponent):
    """
    Jacobian action component which returns the expressions for Jac*x
    """
    def __init__(self, jacobian, result_name="jac_action"):
        """
        Create a JacobianActionComponent
        """
        timer = Timer("Computing jacobian action component")
        check_arg(jacobian, JacobianComponent)
        super(JacobianActionComponent, self).__init__(\
            "JacobianAction", jacobian.root)

        x = self.root.full_state_vector
        jac = jacobian.jacobian

        # Create Jacobian action vector
        self._action_vector = sp.Matrix(len(x), 1,lambda i,j:0)

        self.add_comment("Computing the action of the jacobian")

        self.shapes[result_name] = (len(x),)
        for i, expr in enumerate(jac*x):
            self._action_vector[i] = self.add_indexed_expression(result_name,\
                                                                 i, expr)

        # Get body expressions from parent and extend with jacobian expressions
        self._body_expressions = jacobian.body_expressions[:]
        for obj in self.ode_objects:
            self._body_expressions.append(obj)

    @property
    def action_vector(self):
        """
        Return the vector of all action symbols
        """
        return self._action_vector

class DiagonalJacobianActionComponent(CodeComponent):
    """
    Jacobian action component which returns the expressions for Jac*x
    """
    def __init__(self, diagonal_jacobian):
        """
        Create a DiagonalJacobianActionComponent
        """
        timer = Timer("Computing jacobian action component")
        check_arg(jacobian, DiagonalJacobianComponent)
        super(DiagonalJacobianActionComponent, self).__init__(\
            "DiagonalJacobianAction", diagonal_jacobian.root)

        x = self.root.full_state_vector
        jac = diagonal_jacobian.diagonal_jacobian

        self._action_vector = sp.Matrix(len(x), 1,lambda i,j:0)

        self.add_comment("Computing the action of the jacobian")

        # Create Jacobian matrix
        self.shapes["diag_jac_action"] = (len(x),)
        for i, expr in enumerate(jac*x):
            self._action_vector[i] = self.add_indexed_expression(\
                "diag_jac_action", i, expr)

        # Get body expressions from parent and extend with jacobian expressions
        self._body_expressions = diagonal_jacobian.body_expressions[:]
        self._body_expressions.extend(self.ode_objects)

    @property
    def action_vector(self):
        """
        Return the vector of all action symbols
        """
        return self._action_vector

class FactorizedJacobianComponent(CodeComponent):
    """
    Class to generate expressions for symbolicaly factorizing a jacobian
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

        self.shapes["jac"] = (n,n)
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
    def argument_objects(self):
        """
        Returns a list of ODE objects which are treated as input arguments
        """
        # FIXME:!!!
        return self.jada

    @property
    def num_nonzero(self):
        """
        Return the num non zeros of the Jacobian
        """
        return self._num_nonzero

    @property
    def factorized_jacobian(self):
        return self._factorized_jacobian
        
class ForwardBackwardSubstitutionComponent(CodeComponent):
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
        self.shapes["linearized"] = (self.root.num_full_states,)
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


class ReuseBodyVariableComponent(CodeComponent):
    """
    Class to reuse as much body variables as possible
    """
    def __init__(self, component, *classes):
        timer = Timer("Computing reuse of {0}".format(component.name))
        
        super(ReuseBodyVariableComponent, self).__init__(\
            "ReuseBodyVariables"+component.name, component.root)

        timer = Timer("Substitute reused variables for {0}".format(component.name))

        available_indices = deque()
        body_indices = []
        precent_ind = 0
        max_index = -1
        index_available_at = defaultdict(list)

        # Collect indices
        ind = 0
        replace_dict = {}

        # Create a dummy max shape
        self.shapes["body"] = (len(component.body_expressions),)

        #component.body_expressions
        body_expressions = self._replace_derivatives(component.body_expressions)
        for expr in body_expressions:

            if isinstance(expr, DerivativeExpression):
                error("Reuse of DerivativeExpressions are not "\
                      "implemented: " + expr.name)

            # Skip expr of classes
            if not isinstance(expr, classes):
                body_indices.append(-1)
                if isinstance(expr, Comment):
                    self.add_comment(str(expr))
                    continue

                new_expr = self._recreate_expression(expr, replace_dict)
                replace_dict[expr.sym] = new_expr.sym
                self.ode_objects.append(new_expr)
                
                continue

            # Check if any indices are available at this expression ind
            available_indices.extend(index_available_at[ind])

            # If there are available indices we pick one
            if available_indices:
                precent_ind = available_indices.popleft()
            else:
                max_index += 1
                precent_ind = max_index

            # Store new expression together with the mapping between new
            # and old expression
            replace_dict[expr.sym] = self.add_indexed_expression(\
                "body", precent_ind, expr.expr.xreplace(replace_dict))

            # Check when present ind gets available again
            for used_expr in reversed(self.object_used_in[expr]):
                if used_expr in body_expressions:
                    index_available_at[body_expressions.index(\
                        used_expr)].append(precent_ind)
                    break
            else:
                warning("SHOULD NOT COME HERE!")

            ind += 1
        
        self.shapes["body"] = (max_index,)
        if isinstance(component, IndexedExpressionComponent):
            self.shapes.update(component.shapes)
        self._body_expressions = self.ode_objects[:]

    def _recreate_expression(self, expr, replace_dict):
        """
        Recreate an Expression while applying the replace dict to the expression
        """
        # FIXME: Should we distinguish between the different
        # FIXME: intermediates?
        if isinstance(expr, Intermediate):
            return Intermediate(expr.name, expr.expr.xreplace(replace_dict))

        if isinstance(expr, StateDerivative):
            return StateDerivative(expr.state, expr.expr.xreplace(replace_dict))

        if isinstance(expr, AlgebraicExpression):
            return AlgebraicExpression(expr.state, expr.expr.xreplace(replace_dict))

        if isinstance(expr, IndexedExpression):
            return IndexedExpression(expr.basename, expr.indices,
                                     expr.expr.xreplace(replace_dict))

        error("Should not reach here")

    def _replace_derivatives(self, body_expressions):
        """
        Replace derivatives with intermediates first
        """
        replace_dict = {}
        replaced_der_exprs = {}
        new_body_expressions = []
        present_ode_object = {}

        timer = Timer("Replace derivatives in {0}".format(self.name))
        timer_0 = Timer("Replace derivatives first run")
        
        # First a run to exchange all DerivativeExpressions
        for expr in body_expressions:

            if isinstance(expr, Comment):
                new_body_expressions.append(expr)

            elif isinstance(expr, IndexedExpression):
                present_ode_object[expr.name] = expr
                new_body_expressions.append(expr)
                
            elif isinstance(expr.expr, sp.Number):
                replace_dict[expr.sym] = expr.expr

            elif isinstance(expr.expr, (sp.Symbol, AppliedUndef)):
                name = sympycode(expr.expr)
                replaced_der_exprs[expr] = present_ode_object.get(\
                    name, self.root.present_ode_objects.get(name))
                replace_dict[expr.sym] = expr.expr

            elif isinstance(expr.expr, sp.Mul) and len(expr.expr.args)==2 and \
                     isinstance(expr.expr.args[1], (sp.Symbol, AppliedUndef)) and \
                     expr.expr.args[0]==-sp.S.One:
                name = sympycode(expr.expr.args[1])
                replaced_der_exprs[expr] = present_ode_object.get(\
                    name, self.root.present_ode_objects.get(name))
                replace_dict[expr.sym] = expr.expr

            elif isinstance(expr, DerivativeExpression):
                new_expr = Intermediate(expr.name, expr.expr.xreplace(replace_dict))

                if isinstance(new_expr.expr, sp.Number):
                    replace_dict[expr.sym] = new_expr.expr
                    continue
                
                replaced_der_exprs[expr] = new_expr
                replace_dict[expr.sym] = new_expr.sym
                expr = new_expr
                present_ode_object[expr.name] = expr
                new_body_expressions.append(expr)
                
            else:
                present_ode_object[expr.name] = expr
                new_body_expressions.append(expr)

        new_new_body_expressions = []

        del timer_0

        timer_1 = Timer("Replace derivatives second run")
        # Then a run to exchange all expressions using the exchanged expressions
        replaced_exprs = {}
        all_replaced_der_exprs = replaced_der_exprs.values()
        for expr2 in new_body_expressions:

            # If we have already exchanged this expression
            if expr2 in all_replaced_der_exprs or \
                   isinstance(expr2, Comment):
                new_new_body_expressions.append(expr2)
                continue

            # Check if expr2 contains derivative symbols if so recreate it
            #for expr in replaced_der_exprs:
            #    if expr2.expr.has(expr.sym):
            #        new_expr = self._recreate_expression(expr2, replace_dict)
            #        replaced_exprs[expr2] = new_expr
            #        new_new_body_expressions.append(new_expr)
            #        break
            #else:
            #    new_new_body_expressions.append(expr2)

            new_expr = self._recreate_expression(expr2, replace_dict)
            replaced_exprs[expr2] = new_expr
            new_new_body_expressions.append(new_expr)
            
        del timer_1
        
        timer_2 = Timer("Replace derivatives third run")

        # Merge the replaced expr dicts
        replaced_exprs.update(replaced_der_exprs)
        
        # Update object_used_in dict
        self.object_used_in = {}
        for expr, used_in in self.root.object_used_in.items():
            self.object_used_in[replaced_exprs.get(expr, expr)] = \
                    [replaced_exprs.get(used_expr, used_expr) for used_expr in used_in]

        return new_new_body_expressions
    
