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
           "CommonSubExpressionODE", "CodeComponent", \
           "componentwise_derivative", \
           "linearized_derivatives", "jacobian_expressions", \
           "jacobian_action_expressions", "factorized_jacobian_expressions",
           "forward_backward_subst_expressions",\
           "diagonal_jacobian_expressions", "rhs_expressions"]

# System imports
from collections import OrderedDict, deque, defaultdict
from sympy.core.function import AppliedUndef
import sys

# ModelParameters imports
from modelparameters.sympytools import sp
from modelparameters.codegeneration import sympycode

# Local imports
from gotran.common import error, info, debug, check_arg, check_kwarg, \
     scalars, Timer, warning, tuplewrap, parameters
from utils import ode_primitives
from odeobjects2 import State, Parameter, IndexedObject, Comment
from expressions2 import *
from odecomponents2 import ODEBaseComponent, ODE

#FIXME: Remove our own cse, or move to this module?
from gotran.codegeneration.sympy_cse import cse

def rhs_expressions(ode, result_name="dy"):
    """
    Return a right hand side code component 

    Arguments
    ---------
    ode : ODE
        The finalized ODE for which the ith derivative should be computed
    result_name : str
        The name of the variable storing the rhs result
    """

    check_arg(ode, ODE)
    if not ode.is_finalized:
        error("Cannot compute right hand side expressions if the ODE is "\
              "not finalized")
        
    return CodeComponent("RHSComponent", ode, **{result_name:ode.state_expressions})
    
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
        state), ode, dy=[expr])

def linearized_derivatives(ode, result_name="linearized"):
    """
    Return an ODEComponent holding the linearized derivative expressions

    Arguments
    ---------
    ode : ODE
        The ODE for which derivatives should be linearized
    result_name : str
        The name of the variable storing the linearized derivatives
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return LinearizedDerivativeComponent(ode, result_name)

def jacobian_expressions(ode, result_name="jac"):
    """
    Return an ODEComponent holding expressions for the jacobian

    Arguments
    ---------
    ode : ODE
        The ODE for which the jacobian expressions should be computed
    result_name : str
        The name of the variable storing the jacobian result
    """
    if not ode.is_finalized:
        error("The ODE is not finalized")

    return JacobianComponent(ode, result_name=result_name)

def diagonal_jacobian_expressions(jacobian, result_name="diag_jac",\
                                  jacobian_name="jac"):
    """
    Return an ODEComponent holding expressions for the diagonal jacobian

    Arguments
    ---------
    jacobian : JacobianComponent
        The Jacobian of the ODE
    result_name : str
        The name of the variable storing the jacobian diagonal result
    jacobian_name : str (optional)
        The basename of the jacobian name used in the jacobian component
    """
    return DiagonalJacobianComponent(jacobian, result_name, jacobian_name)

def jacobian_action_expressions(jacobian, with_body=True, \
                                result_name="diag_jac_action",\
                                jacobian_name="jac"):
    """
    Return an ODEComponent holding expressions for the jacobian action

    Arguments
    ---------
    jacobian : JacobianComponent
        The ODEComponent holding expressions for the jacobian
    with_body : bool
        If true, the body for computing the jacobian will be included 
    result_name : str
        The name of the variable storing the jacobian diagonal result
    jacobian_name : str (optional)
        The basename of the jacobian name used in the jacobian component
    """
    
    check_arg(jacobian, JacobianComponent)
    return JacobianActionComponent(jacobian, with_body, result_name, 
                                   jacobian_name)

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

class CodeComponent(ODEBaseComponent):
    """
    A wrapper class around an ODE. Its primary purpose is to help
    generate code.

    
    
    
    which alows extraction and manipulation of the ODE expressions.

    n ODEComponent which allows adding indexed expressions.

    The class is a wrapper around a
    """
    def __init__(self, name, ode, **results):
        """
        Creates a CodeComponent

        Arguments
        ---------
        name : str
            The name of the component. This str serves as the unique
            identifier of the Component.
        ode : ODE
            The parent component which need to be a ODE
        results : kwargs
            A dict of result expressions. The result expressions will
            be used to extract the body expressions for the code component.
        """
        check_arg(ode, ODE)
        super(CodeComponent, self).__init__(name, ode)
        
        for result_name, result_expressions in results.items():
            check_kwarg(result_expressions, result_name, list, \
                        itemtypes=(Expression, Comment))
        
        # Shapes for any indexed expressions or objects
        self.shapes = {}

        # A map between expressions and recreated IndexedExpressions
        self.indexed_map = defaultdict(OrderedDict)

        # Init parameter or state replace dict
        self._init_param_state_replace_dict()

        # Recreate body expressions based on the given result_expressions
        if results:
            results, body_expressions = self._body_from_results(**results)
            self.body_expressions = self._recreate_body(\
                body_expressions, **results)
        else:
            self.body_expressions = []
            
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
                  "shape for {1}{2}".format(indices, basename, self.shapes[basename]))

        for dim, (index, shape_ind) in enumerate(zip(indices, self.shapes[basename])):
            if index >= shape_ind:
                error("Indices must be smaller or equal to the shape. Missmatch "\
                      "in dim {0}: {1}>={2}".format(dim+1, index, shape_ind))

        # Create the indexed expression
        expr = IndexedExpression(basename, indices, expr, self.shapes[basename])
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

        # Check that provided indices fit with the registered shape
        if len(self.shapes[basename]) > len(indices):
            error("Shape missmatch between indices {0} and registered "\
                  "shape for {1}{2}".format(indices, basename, self.shapes[basename]))

        for dim, (index, shape_ind) in enumerate(zip(indices, self.shapes[basename])):
            if index >= shape_ind:
                error("Indices must be smaller or equal to the shape. Missmatch "\
                      "in dim {0}: {1}>={2}".format(dim+1, index, shape_ind))

        # Create IndexedObject
        obj = IndexedObject(basename, indices, self.shapes[basename])
        self._register_component_object(obj)

        # Return the sympy version of the object
        return obj.sym

        
    def indexed_objects(self, *basenames):
        """
        Return a list of all indexed objects with the given basename,
        if no base names give all indexed objects are returned
        """
        if not basenames:
            basenames = self.shapes.keys()
        return [obj for obj in self.ode_objects if isinstance(\
            obj, IndexedObject) and obj.basename in basenames]

    def _recreate_expression(self, expr, der_replace_dict, replace_dict):
        """
        Recreate an Expression while applying first the replace dict
        containg all derivatives and then a replace dict containing all the rest
        """

        # First do the replacements
        sympyexpr = expr.expr.xreplace(der_replace_dict).xreplace(replace_dict)
        
        # FIXME: Should we distinguish between the different
        # FIXME: intermediates?
        if isinstance(expr, Intermediate):
            new_expr = Intermediate(expr.name, sympyexpr)

        elif isinstance(expr, StateDerivative):
            new_expr = StateDerivative(expr.state, sympyexpr)

        elif isinstance(expr, AlgebraicExpression):
            new_expr = AlgebraicExpression(expr.state, sympyexpr)

        elif isinstance(expr, IndexedExpression):
            new_expr = IndexedExpression(expr.basename, expr.indices, \
                                         sympyexpr, self.shapes[expr.basename])
        else:
            error("Should not reach here")

        # Inherit the count of the old expression
        new_expr._recount(expr._count)

        return new_expr

    def _init_param_state_replace_dict(self):
        """
        Create a parameter state replace dict based on the values in the
        global parameters 
        """
        
        param_state_replace_dict = {}

        param_repr = parameters["code_generation"]["parameters"]["representation"]
        param_name = parameters["code_generation"]["parameters"]["array_name"]
        
        state_repr = parameters["code_generation"]["states"]["representation"]
        state_name = parameters["code_generation"]["states"]["array_name"]

        # Create a map between states, parameters 
        state_param_map = dict(states=OrderedDict(\
            (state, IndexedObject(state_name, ind)) \
            for ind, state in enumerate(self.root.full_states)),
                               parameters=OrderedDict(\
                                   (param, IndexedObject(param_name, ind)) \
                                   for ind, param in enumerate(\
                                       self.root.parameters)))
        
        # If not having named parameters
        if param_repr == "numerals":
            param_state_replace_dict.update((param.sym, param.init) for \
                                            param in self.root.parameters)
        elif param_repr == "array":
            self.shapes[param_name] = (self.root.num_parameters,)
            
            param_state_replace_dict.update((state.sym, indexed.sym) \
                                            for state, indexed in \
                                            state_param_map["states"].items())

        if state_repr == "array":
            self.shapes[state_name] = (self.root.num_full_states,)
            param_state_replace_dict.update((param.sym, indexed.sym) \
                                            for param, indexed in \
                                            state_param_map["parameters"].items())

        # Store dicts
        self.param_state_replace_dict = param_state_replace_dict
        self.indexed_map.update(state_param_map)

    def _body_from_results(self, **results):
        """
        Returns a sorted list of body expressions all used in the result expressions
        """

        if not results:
            return {}, []

        # Check if we are using common sub expressions for body
        if parameters["code_generation"]["body"]["use_cse"]:
            return self._body_from_cse(**results)
        else:
            return self._body_from_dependencies(**results)

    def _body_from_cse(self, **results):
        
        timer = Timer("Compute common sub expressions for {0}".format(self.name))

        # Extract all result expressions
        orig_result_expressions = reduce(sum, (result_exprs for result_exprs in \
                                               results.values()), [])
        
        # A map between result expression and result name
        result_names = dict((result_expr, result_name) for \
                            result_name, result_exprs in results.items() \
                            for result_expr in result_exprs)

        # The expanded result expressions
        expanded_result_exprs = [self.root.expanded_expressions[obj.name] \
                                 for obj in orig_result_expressions]

        # Collect results and body_expressions
        body_expressions = []
        new_results = defaultdict(list)

        # Set shape for result expressions
        for result_name, result_expressions in results.items():
            if result_name not in self.shapes:
                self.shapes[result_name] = (len(result_expressions),)

        might_take_time = len(orig_result_expressions) >= 40

        if might_take_time:
            info("Computing common sub expressions for {0}. Might take "\
                 "some time...".format(self.name))
            sys.stdout.flush()

        # Call sympy common sub expression reduction
        cse_exprs, cse_result_exprs = cse(expanded_result_exprs,\
                                          symbols=sp.numbered_symbols("cse_"),\
                                          optimizations=[])
        cse_cnt = 0
        cse_subs = {}
        atoms = [state.sym for state in self.root.full_states]
        atoms.extend(param.sym for param in self.root.parameters)

        # Register the common sub expressions as Intermediates
        for sub, expr in cse_exprs:
        
            # If the expression is just one of the atoms of the ODE we skip
            # the cse expressions but add a subs for the atom
            if expr in atoms:
                cse_subs[sub] = expr
            else:

                # Add body expression as an intermediate expression
                sym = self.add_intermediate("cse_{0}".format(\
                    cse_cnt), expr.xreplace(cse_subs))
                cse_subs[sub] = sym
                cse_cnt += 1
                body_expressions.append(self.ode_objects.get(sympycode(sym)))

        # Register the state expressions
        for ind, (orig_result_expr, result_expr) in \
                enumerate(zip(orig_result_expressions, cse_result_exprs)):

            result_name = result_names[orig_result_expr]

            # Replace pure state and param expressions
            exp_expr = result_expr.xreplace(cse_subs)

            # Add result expression as an indexed expression, if original
            # expression already is an IndexedExpression then recreate the
            # Expression using the same Indices
            if isinstance(orig_result_expr, IndexedExpression):
                sym = self.add_indexed_expression(result_name, \
                                                  orig_result_expr.indices, exp_expr)
            else:
                sym = self.add_indexed_expression(result_name, ind, exp_expr)
            expr = self.ode_objects.get(sympycode(sym))

            # Register the new result expression
            new_results[result_name].append(expr)
            body_expressions.append(expr)

        if might_take_time:
            info(" done")

        return new_results, body_expressions
        
    def _body_from_dependencies(self, **results):
        
        timer = Timer("Compute dependencies for {0}".format(self.name))
        
        # Extract all result expressions
        result_expressions = reduce(sum, (result_exprs for result_exprs in \
                                          results.values()), [])
        
        # Check passed expressions
        ode_expr_deps = self.root.expression_dependencies
        exprs = set(result_expressions)
        not_checked = set()
        used_states = set()
        used_parameters = set()

        exprs_not_in_body = []

        for expr in result_expressions:
            check_arg(expr, (Expression, Comment), \
                      context=CodeComponent._body_from_results)
            
            if isinstance(expr, Comment):
                continue

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
        self.used_states = sorted(used_states)
        self.used_parameters = sorted(used_parameters)

        # Return a sorted list of all collected expressions
        return results, sorted(list(exprs))

    def _recreate_body(self, body_expressions, **results):
        """
        Create body expressions based on the given result_expressions
        
        In this method are all expressions replaced with something that should
        be used to generate code. The parameters in:

            parameters["code_generation"]

        decides how parameters, states, body expressions and indexed expressions
        are represented.
        
        """

        if not (results or body_expressions):
            return 
            
        for result_name, result_expressions in results.items():
            check_kwarg(result_expressions, result_name, list, \
                        context=CodeComponent._recreate_body, \
                        itemtypes=(Expression, Comment))

        # Extract all result expressions
        result_expressions = reduce(sum, (result_exprs for result_exprs in \
                                          results.values()), [])

        # A map between result expression and result name
        result_names = dict((result_expr, result_name) for \
                            result_name, result_exprs in results.items() \
                            for result_expr in result_exprs)

        timer = Timer("Recreate body expressions for {0}".format(self.name))
        
        # Initialize the replace_dictionaries
        replace_dict = self.param_state_replace_dict
        der_replace_dict = {}

        # Get a copy of the map of where objects are used in and their
        # present dependencies so any updates done in these dictionaries does not
        # affect the original dicts
        object_used_in = dict((expr, used.copy()) for expr, used in \
                              self.root.object_used_in.items())

        expression_dependencies = dict((expr, deps.copy())for expr, deps in \
                                       self.root.expression_dependencies.items())
        
        # Get body parameters
        body_repr = parameters["code_generation"]["body"]["representation"]
        optimize_exprs = parameters["code_generation"]["body"]["optimize_exprs"]

        # Set body related variables if the body should be represented by an array
        if "array" in body_repr:
            body_name = parameters["code_generation"]["body"]["array_name"]
            available_indices = deque()
            body_indices = []
            max_index = -1
            body_ind = 0
            index_available_at = defaultdict(list)
            if body_name == result_name:
                error("body and result cannot have the same name.")

            # Initiate shapes with inf
            self.shapes[body_name] = (float("inf"),)

        # Iterate over body expressions and recreate the different expressions
        # according to state, parameters, body and result expressions
        replaced_expr_map = OrderedDict()
        new_body_expressions = []
        present_ode_objects = dict((state.name, state) for state in self.root.full_states)
        present_ode_objects.update((param.name, param) for param in self.root.parameters)

        def store_expressions(expr, new_expr):
            "Help function to store new expressions"
            
            # Update sym replace dict
            if isinstance(expr, Derivatives):
                der_replace_dict[expr.sym] = new_expr.sym
            else:
                replace_dict[expr.sym] = new_expr.sym

            # Store the new expression for later references
            present_ode_objects[expr.name] = new_expr
            replaced_expr_map[expr] = new_expr

            # Append the new expression
            new_body_expressions.append(new_expr)

            # Update dependency information
            if expr in object_used_in:
                for dep in object_used_in[expr]:
                    expression_dependencies[dep].remove(expr)
                    expression_dependencies[dep].add(new_expr)
                object_used_in[new_expr] = object_used_in.pop(expr)

            if expr in expression_dependencies:
                expression_dependencies[new_expr] = expression_dependencies.pop(\
                    expr)

        # The main iteration over all body_expressions
        for expr in body_expressions:

            # 1) Comments
            if isinstance(expr, Comment):
                new_body_expressions.append(expr)
                continue

            assert(isinstance(expr, Expression))

            # 2) Check for expression optimzations 
            if not (optimize_exprs == "none" or expr in result_expressions):
                
                # If expr is just a number we exchange the expression with the
                # number
                if "numerals" in optimize_exprs and \
                       isinstance(expr.expr, sp.Number):
                    replace_dict[expr.sym] = expr.expr

                    # Remove information about this expr beeing used
                    for dep in object_used_in[expr]:
                        expression_dependencies[dep].remove(expr)
                    object_used_in.pop(expr)
                    continue

                # If the expr is just a symbol (symbol multiplied with a scalar)
                # we exchange the expression with the sympy expressions
                elif "symbols" in  optimize_exprs and \
                       (isinstance(expr.expr, (sp.Symbol, AppliedUndef)) or \
                        isinstance(expr.expr, sp.Mul) and len(expr.expr.args)==2 and \
                        isinstance(expr.expr.args[1], (sp.Symbol, AppliedUndef)) and \
                        expr.expr.args[0].is_number):

                    # Add a replace rule based on the stored sympy expression
                    sympy_expr = expr.expr.xreplace(der_replace_dict).xreplace(\
                        replace_dict)
                    
                    if isinstance(expr.sym, sp.Derivative):
                        der_replace_dict[expr.sym] = sympy_expr
                    else:
                        replace_dict[expr.sym] = sympy_expr
                        
                    # Get exchanged repr
                    if isinstance(expr.expr, (sp.Symbol, AppliedUndef)):
                        name = sympycode(expr.expr)
                    else:
                        name = sympycode(expr.expr.args[1])

                    dep_expr = present_ode_objects[name]

                    # If using reused body expressions we need to update the
                    # index information so that the index previously available
                    # for this expressions gets available at the last expressions
                    # the present expression is used in.
                    if "reused" in body_repr:
                        available_ind = index_available_at[expr]

                        # Update when ind will be available
                        if available_ind:
                            for used_expr in reversed(list(object_used_in[expr])):
                                if used_expr in body_expressions:
                                    index_available_at[used_expr].append(\
                                        available_ind[0])
                                    break
                    
                    # Update information about this expr beeing used
                    for dep in object_used_in[expr]:
                        expression_dependencies[dep].remove(expr)
                        expression_dependencies[dep].add(dep_expr)
                    
                    object_used_in.pop(expr)
                    continue

            # 3) General operations for all Expressions that are kept

            # Before we process the expression we check if any indices gets
            # available with the expr (Only applies for the "reused" option for
            # body_repr.)
            if "reused" in body_repr:

                # Check if any indices are available at this expression ind
                available_indices.extend(index_available_at[expr])
                    
            # Store a map of old name this will preserve the ordering of
            # expressions with the same name, similar to how this is treated in
            # the actuall ODE.
            present_ode_objects[expr.name] = expr

            # 4) Handle result expression
            if expr in result_expressions:

                # Get the result name
                result_name = result_names[expr]
                
                # If the expression is an IndexedExpression with the same basename
                # as the result name we just recreate it
                if isinstance(expr, IndexedExpression) and \
                       result_name == expr.basename:
                    
                    new_expr = self._recreate_expression(\
                        expr, der_replace_dict, replace_dict)

                # Not an indexed expression
                else:
                    
                    # Get index based on the original ordering
                    index = (result_expressions.index(expr),)

                    # Create the IndexedExpression
                    # NOTE: First replace any derivative expression replaces, then state and
                    # NOTE: params
                    new_expr = IndexedExpression(result_name, index, expr.expr.\
                                                 xreplace(der_replace_dict).\
                                                 xreplace(replace_dict))

                    self.indexed_map[new_expr.basename][expr] = new_expr

                    # Copy counter from old expression so it sort properly
                    new_expr._recount(expr._count)

                # Store the expressions
                store_expressions(expr, new_expr)
                    
            # 4) Handle indexed expression
            # All indexed expressions are just kept but recreated with updated
            # sympy expressions
            elif isinstance(expr, IndexedExpression):

                new_expr = self._recreate_expression(expr, der_replace_dict, \
                                                     replace_dict)
                
                # Store the expressions
                store_expressions(expr, new_expr)
            

            # 5) If replacing all body exressions with an indexed expression
            elif "array" in body_repr:
                
                # 5a) If we reuse array indices
                if "reused" in body_repr:
                    
                    if available_indices:
                        ind = available_indices.popleft()
                    else:
                        max_index += 1
                        ind = max_index

                    # Check when present ind gets available again
                    for used_expr in reversed(list(object_used_in[expr])):
                        if used_expr in body_expressions:
                            index_available_at[used_expr].append(ind)
                            break
                    else:
                        warning("SHOULD NOT COME HERE!")
                        
                # 5b) No reuse of array indices. Here each index corresponds to
                #     a distinct body expression
                else:

                    ind = body_ind

                    # Increase body_ind
                    body_ind += 1
                    
                # Create the IndexedExpression
                new_expr = IndexedExpression(body_name, ind, expr.expr.\
                                             xreplace(der_replace_dict).\
                                             xreplace(replace_dict))

                self.indexed_map[body_name][expr] = new_expr

                # Copy counter from old expression so they sort properly
                new_expr._recount(expr._count)
                
                # Store the expressions
                store_expressions(expr, new_expr)
                
            # 6) If the expression is just and ordinary body expression 
            else:

                new_expr = self._recreate_expression(expr, der_replace_dict, replace_dict)
                
                # Store the expressions
                store_expressions(expr, new_expr)

        # Store indices for any added arrays
        if "reused_array" == body_repr:

            self.shapes[body_name] = (max_index,)

        elif "array" == body_repr:

            self.shapes[body_name] = (body_ind,)

        # Store the shape of the added result expressions
        for result_name, result_expressions in results.items():
            if result_name not in self.shapes:
                self.shapes[result_name] = (len(result_expressions),)

        return new_body_expressions
        
class JacobianComponent(CodeComponent):
    """
    An ODEComponent which keeps all expressions for the Jacobian of the rhs
    """
    def __init__(self, ode, result_name="jac"):
        """
        Create a JacobianComponent

        Arguments
        ---------
        ode : ODE
            The parent component of this ODEComponent
        """
        check_arg(ode, ODE)

        # Call base class using empty result_expressions
        super(JacobianComponent, self).__init__("Jacobian", ode)

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
    def __init__(self, jacobian, result_name="diag_jac", jacobian_name="jac"):
        """
        Create a DiagonalJacobianComponent

        Arguments
        ---------
        jacobian : JacobianComponent
            The Jacobian of the ODE
        result_name : str (optional)
            The basename of the indexed result expression
        jacobian_name : str (optional)
            The basename of the jacobian name used in the jacobian component
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
    def __init__(self, jacobian, with_body=True, result_name="jac_action", \
                 jacobian_name="jac"):
        """
        Create a JacobianActionComponent

        Arguments
        ---------
        jacobian : JacobianComponent
            The Jacobian of the ODE
        with_body : bool
            If true, the body for computing the jacobian will be included 
        result_name : str
            The basename of the indexed result expression
        jacobian_name : str (optional)
            The basename of the jacobian name used in the jacobian component
        """
        timer = Timer("Computing jacobian action component")
        check_arg(jacobian, JacobianComponent)
        super(JacobianActionComponent, self).__init__(\
            "JacobianAction", jacobian.root)

        x = self.root.full_state_vector
        jac = jacobian.jacobian

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
                 result_name="diag_jac_action"):
        """
        Create a DiagonalJacobianActionComponent

        Arguments
        ---------
        jacobian : JacobianComponent
            The Jacobian of the ODE
        with_body : bool
            If true, the body for computing the jacobian will be included 
        result_name : str
            The basename of the indexed result expression
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
    def __init__(self, jacobian, jacobian_name="jac"):
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
    def __init__(self, ode, result_name="linearized"):

        super(LinearizedDerivativeComponent, self).__init__(\
            "LinearizedDerivatives", ode)
        
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


