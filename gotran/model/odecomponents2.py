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

__all__ = ["DerivativeComponent", "ReactionComponent", "MarkovModelComponent",\
           "ODE"]

# System imports
from collections import OrderedDict, defaultdict
import re
import weakref

from sympy.core.function import AppliedUndef

# ModelParameters imports
from modelparameters.sympytools import sp, symbols_from_expr
from modelparameters.codegeneration import sympycode
from modelparameters.utils import Timer

# Local imports
from gotran.common import error, debug, check_arg, check_kwarg, scalars
from gotran.model.odeobjects2 import *
from gotran.model.expressions2 import *
from odebasecomponent import *

_derivative_name_template = re.compile("\Ad([a-zA-Z]\w*)_d([a-zA-Z]\w*)\Z")
_algebraic_name_template = re.compile("\Aalg_([a-zA-Z]\w*)_0\Z")
_rate_name_template = re.compile("\Arate_([a-zA-Z]\w*)_([a-zA-Z]\w*)\Z")

# Flags for determine special expressions
INTERMEDIATE = 0
ALGEBRAIC_EXPRESSION = 1
DERIVATIVE_EXPRESSION = 2
RATE_EXPRESSION = 3
STATE_SOLUTION_EXPRESSION = 4

special_expression_str = {
    INTERMEDIATE : "intermediate expression",
    ALGEBRAIC_EXPRESSION : "algebraic expression",
    DERIVATIVE_EXPRESSION : "derivative expression",
    RATE_EXPRESSION : "rate expression",
    STATE_SOLUTION_EXPRESSION : "state solution expression",
    }

def special_expression(name, root):
    """
    Check if an expression name corresponds to a special expression
    """

    alg_expr = re.search(_algebraic_name_template, name)
    if alg_expr:
        return alg_expr, ALGEBRAIC_EXPRESSION

    der_expr = re.search(_derivative_name_template, name)
    if der_expr:
        return der_expr, DERIVATIVE_EXPRESSION

    rate_expr = re.search(_rate_name_template, name)
    if rate_expr:
        return rate_expr, RATE_EXPRESSION

    state_comp = root.present_ode_objects.get(name)
    if state_comp and isinstance(state_comp.obj, State):
        return state_comp.obj, STATE_SOLUTION_EXPRESSION

    return None, INTERMEDIATE

class DerivativeComponent(ODEBaseComponent):
    """
    ODE Component for derivative and algebraic expressions
    """

    def __init__(self, name, parent):
        """
        Create a DerivativeComponent

        Arguments
        ---------
        name : str
            The name of the component. This str serves as the unique
            identifier of the Component.
        parent : ODEBaseComponent
            The parent component of this ODEComponent
        """
        super(DerivativeComponent, self).__init__(name, parent)

    def __setattr__(self, name, value):
        """
        A magic function which will register expressions and simpler
        state expressions
        """

        # If we are registering a protected attribute or an attribute
        # during construction, just add it to the dict
        if name[0] == "_" or not self._constructed:
            self.__dict__[name] = value
            return

        # If no expression is registered
        if (not isinstance(value, scalars+(sp.Number,))) \
               and not (isinstance(value, sp.Basic) and symbols_from_expr(value)):
            debug("Not registering: {0} as attribut. It does not contain "\
                  "any symbols or scalars.".format(name))

            # FIXME: Should we raise an error?
            return

        # Check for special expressions
        expr, TYPE = special_expression(name, self.root)
        
        if TYPE == INTERMEDIATE:
            self.add_intermediate(name, value)

        elif TYPE == DERIVATIVE_EXPRESSION:

            # Try getting corresponding ODEObjects
            expr_name, var_name = expr.groups()
            expr_obj = self.root.present_ode_objects.get(expr_name)
            var_obj = self.root.present_ode_objects.get(var_name)

            # If the expr or variable is not declared in this ODE we
            # register an intermediate
            if expr_obj is None or var_obj is None:
                self.add_intermediate(name, value)
                
            #if expr_obj is None:
            #    error("Trying to register a DerivativeExpression, but "\
            #          "the expression: '{0}' is not registered in this "\
            #          "ODE.".format(expr_name))
            #
            #if var_obj is None:
            #    error("Trying to register a DerivativeExpression, but "\
            #          "the variable: '{0}' is not registered in this "\
            #          "ODE.".format(var_name))
            else:
                self.add_derivative(expr_obj.obj, var_obj.obj, value)

        elif TYPE == STATE_SOLUTION_EXPRESSION:
            self.add_state_solution(expr, value)

        elif TYPE == ALGEBRAIC_EXPRESSION:

            # Try getting corresponding ODEObjects
            var_name = expr.groups()
            var_obj = self.root.present_ode_objects.get(var_name)

            if var_obj is None:
                self.add_intermediate(name, value)
            else:
                self.add_algebraic(var_obj, expr)
            
        else:
            error("Trying to register a {0} but that is not allowed in a"\
                  "Derivative component.".format(special_expression_str[TYPE]))

class ReactionComponent(ODEBaseComponent):
    """
    A class for a special type of state derivatives
    """
    def __init__(self, name, parent, volume, species):
        """
        Create an ReactionComponent

        Arguments
        ---------
        name : str
            The name of the component. This str serves as the unique
            identifier of the Component.
        parent : ODEBaseComponent
            The parent component of this ODEComponent
        """
        raise NotImplementedError("ReactionComponent is not implemented")
        super(ReactionComponent, self).__init__(name, parent)

    def add_reaction(self, reactants, products, expr):
        pass

class MarkovModelComponent(ODEBaseComponent):
    """
    A class for a special type of state derivatives
    """
    def __init__(self, name, parent):
        """
        Create an ReactionComponent

        Arguments
        ---------
        name : str
            The name of the component. This str serves as the unique
            identifier of the Component.
        parent : ODEBaseComponent
            The parent component of this ODEComponent
        """
        super(MarkovModelComponent, self).__init__(name, parent)
        
        # Rate attributes
        self._rates = OrderedDict()

    def add_rates(self, states, rate_matrix):
        """
        Use a rate matrix to set rates between states

        Arguments
        ---------
        states : list of States, tuple of two lists of States
            If one list is passed the rates should be a square matrix
            and the states list determines the order of the row and column of
            the matrix. If two lists are passed the first determines the states
            in the row and the second the states in the column of the Matrix
        rates_matrix : sympy.MatrixBase
            A sympy.Matrix of the rate expressions between the states given in
            the states argument
        """

        check_arg(states, (tuple, list), 0, MarkovModelComponent.add_rates)
        check_arg(rate_matrix, sp.MatrixBase, 1, MarkovModelComponent.add_rates)

        # If list
        if isinstance(states, list):
            states = (states, states)

        # else tuple
        elif len(states) != 2 and not all(isinstance(list_of_states, list) \
                                          for list_of_states in states):
            error("expected a tuple of 2 lists with states as the "\
                  "states argument")

        # Get all states associated with this Markov model
        local_states = self.states
        
        # Check index arguments
        for list_of_states in states:

            if not all(state in local_states for state in list_of_states):
                error("Expected the states arguments to be States in "\
                      "the Markov model")

        # Check that the length of the state lists corresponds with the shape of
        # the rate matrix
        if rate.shape[0] != len(states[0]) or rate.shape[1] != len(states[1]):
            error("Shape of rates does not match given states")

        for i, state_i in enumerate(states[0]):
            for j, state_j in enumerate(states[1]):
                value = rate[i,j]

                # If 0 as rate
                if (isinstance(value, scalars) and value == 0) or \
                    (isinstance(value, sp.Basic) and value.is_zero):
                    continue

                if state_i == state_j:
                    error("Cannot have a nonzero rate value between the "\
                          "same states")

                # Assign the rate
                self.add_single_rate(state_i, state_j, value)

    def add_single_rate(self, to_state, from_state, expr):
        """
        Add a single rate expression
        """
        
        check_arg(expr, scalars + (sp.Basic,), 2, \
                  MarkovModelComponent.add_single_rate)

        expr = sp.sympify(expr)
        
        to_state = self._expect_state(to_state, \
                                      allow_state_solution=True)
        from_state = self._expect_state(from_state, \
                                        allow_state_solution=True)

        if to_state == from_state:
            error("The two states cannot be the same.")

        if (sympycode(to_state), sympycode(from_state)) in self._rates:
            error("Rate between state {0} and {1} is already "\
                  "registered.".format(from_state, to_state))

        if to_state.sym in expr or from_state.sym in expr:
            error("The rate expression cannot be dependent on the "\
                  "states it connects.")

        # Create a RateExpression
        obj = RateExpression(to_state, from_state, expr)

        self._register_component_object(obj)

        self._rates[sympycode(to_state), sympycode(from_state)] = obj

    def __setattr__(self, name, value):
        """
        A magic function which will register intermediates and rate expressions
        """

        # If we are registering a protected attribute or an attribute
        # during construction, just add it to the dict
        if name[0] == "_" or not self._constructed:
            self.__dict__[name] = value
            return

        # If no expression is registered
        if (not isinstance(value, scalars)) and not (isinstance(value, sp.Basic) \
                                                     and symbols_from_expr(value)):
            debug("Not registering: {0} as attribut. It does not contain "\
                  "any symbols or scalars.".format(name))

            # FIXME: Should we raise an error?
            return


        # Check for special expressions
        expr, TYPE = special_expression(name, self.root)
        
        if TYPE == INTERMEDIATE:
            self.add_intermediate(name, value)

        elif TYPE == RATE_EXPRESSION:
            to_state_name, from_state_name = rate_expr.groups()

            to_state = self.ode_objects.get(to_state_name)
            if not to_state:
                error("Trying to register a rate expression but '{0}' is "\
                      "not a state in this Markov model.".format(\
                          to_state_name))
            from_state = self.ode_objects.get(from_state_name)
            
            if not from_state:
                error("Trying to register a rate expression but '{0}' is "\
                      "not a state in this Markov model.".format(\
                          from_state_name))
            
            self.add_single_rate(to_state, from_state, value)
        
        elif TYPE == STATE_SOLUTION_EXPRESSION:
            self.add_state_solution(expr, value)

        else:
            error("Trying to register a {0} but that is not allowed in a"\
                  "Markov model component.".format(special_expression_str[TYPE]))

    def finalize_component(self):
        """
        Finalize the Markov model.

        This will add the derivatives to the ode model. After this is
        done no more rates can be added to the Markov model.
        """
        if self._is_finalized:
            error("Cannot finalize a component that is already finalized")

        # Derivatives
        states = self.states
        derivatives = defaultdict(lambda : sp.sympify(0.0))
        rate_check = defaultdict(lambda : 0)

        # Build rate information and check that each rate is added in a
        # symetric way
        used_states = [0]*self.num_states
        for (from_state, to_state), rate in self._rates.items():

            # Get ODEObjects
            from_state = self.ode_objects.get(from_state)
            to_state = self.ode_objects.get(to_state)

            # Add to derivatives of the two states
            derivatives[from_state] -= rate.sym*from_state.sym
            derivatives[to_state] += rate.sym*from_state.sym
            
            if isinstance(from_state, StateSolution):
                from_state = from_state.state

            if isinstance(to_state, StateSolution):
                to_state = to_state.state
            
            # Register rate
            ind_from = states.index(from_state)
            ind_to = states.index(to_state)
            ind_tuple = (min(ind_from, ind_to), max(ind_from, ind_to))
            rate_check[ind_tuple] += 1

            used_states[ind_from] = 1
            used_states[ind_to] = 1

        # Check used states
        if 0 in used_states:
            error("No rate registered for state {0}".format(\
                states[used_states.find(0)]))

        # Check rate symetry
        for (ind_from, ind_to), times in rate_check.items():
            if times != 2:
                error("Only one rate between the states {0} and {1} was "\
                      "registered, expected two.".format(\
                          states[ind_from], states[ind_to]))

        # Add derivatives
        for state in states:
            
            # Skip solved states
            if not isinstance(state, State) or state.is_solved:
                continue
            
            self.add_derivative(state, state.time.sym, derivatives[state])
            #obj = StateDerivative(state, derivatives[state])
            #self._register_component_object(obj)

        assert self.is_locally_complete, "The Markov model should be complete..."
            
        self._is_finalized = True

class ODE(DerivativeComponent):
    """
    Root ODEComponent

    Arguments:
    ----------
    name : str
        The name of the ODE
    ns : dict (optional)
        A namespace which will be filled with declared ODE symbols
    """


    def __new__(cls, *args, **kwargs):
        self = object.__new__(cls, *args, **kwargs)
        return self
    
    def __init__(self, name, ns=None):

        # Call super class with itself as parent component
        super(ODE, self).__init__(name, self)

        # Reset constructed attribute
        self._constructed = False

        # If namespace provided just keep a weak ref
        if ns is None:
            self._ns = {}
        else:
            self._ns = weakref.ref(ns) 

        # Add Time object
        # FIXME: Add information about time unit dimensions and make
        # FIXME: it possible to have different time names
        time = Time("t", "ms")
        self._time = time
        self.ode_objects.append(time)

        # Namespace, which can be used to eval an expression
        self.ns.update({"t":time.sym, "time":time.sym})
        
        # An list with all component names with expression added to them
        # The components are always sorted wrt last expression added
        self.all_expr_components_ordered = []

        # A dict with all components objects
        self.all_components = weakref.WeakValueDictionary()
        self.all_components[name] = self

        # An attribute keeping track of the present ODE component
        self._present_component = self.name

        # A dict with the present ode objects
        # NOTE: hashed by name so duplicated expressions are not stored
        self.present_ode_objects = {}
        self.present_ode_objects["t"] = PresentObjTuple(self._time, self)
        self.present_ode_objects["time"] = PresentObjTuple(self._time, self)
        #self.present_ode_objects = dict(t=(self._time, self), time=(self._time, self))

        # Keep track of duplicated expressions
        self.duplicated_expressions = defaultdict(list)

        # Keep track of expression dependencies and in what expression
        # an object has been used in
        self.expression_dependencies = defaultdict(set)
        self.object_used_in = defaultdict(set)

        # All expanded expressions
        self.expanded_expressions = dict()

        # Attributes which will be populated later
        self._body_expressions = None
        self._mass_matrix = None
        
        # Global finalized flag
        self._is_finalized_ode = False
        
        # Flag that the ODE is constructed
        self._constructed = True

    @property
    def ns(self):
        if isinstance(self._ns, dict):
            return self._ns

        # Check if ref in weakref is alive
        ns = self._ns()
        if isinstance(ns, dict):
            return ns

        # If not just return an empty dict
        return {}

    @property
    def present_component(self):
        """
        Return the present component
        """
        return self.all_components[self._present_component]

    def add_sub_ode(self, subode, prefix=None, components=None,
                   skip_duplicated_global_parameters=True):
        """
        Load an ODE and add it to the present ODE

        Argument
        --------
        subode : str, ODE
            The subode which should be added. If subode is a str an
            ODE stored in that file will be loaded. If it is an ODE it will be
            added directly to the present ODE.
        prefix : str (optional)
            A prefix which all state and parameters are prefixed with. If not
            given the name of the subode will be used as prefix. If set to
            empty string, no prefix will be used.
        components : list, tuple of str (optional)
            A list of components which will be extracted and added to the present
            ODE. If not given the whole ODE will be added.
        skip_duplicated_global_parameters : bool (optional)
            If true global parameters and variables will be skipped if they exists
            in the present model.
        """
        pass

    def register_ode_object(self, obj, comp):
        """
        Register an ODE object in the root ODEComponent
        """

        if self._is_finalized_ode and isinstance(obj, StateExpression):
            error("Cannot register a StateExpression, the ODE is finalized")

        # Check for existing object in the ODE
        duplication = self.present_ode_objects.get(obj.name)

        # If object with same name is already registered in the ode we
        # need to figure out what to do
        if duplication:

            dup_obj, dup_comp = duplication.obj, duplication.comp

            # If State, Parameter or DerivativeExpression we always raise an error
            if isinstance(dup_obj, State) and isinstance(obj, StateSolution):
                debug("Reduce state '{0}' to {1}".format(dup_obj, obj.expr))

            elif any(isinstance(oo, (State, Parameter, Time, DerivativeExpression,
                                     AlgebraicExpression, StateSolution)) \
                     for oo in [dup_obj, obj]):
                error("Cannot register {0}. A {1} with name '{2}' is "\
                      "already registered in this ODE.".format(\
                          type(obj).__name__, type(\
                              dup_obj).__name__, dup_obj.name))
            else:

                # Sanity check that both obj and dup_obj are Expressions
                assert all(isinstance(oo, (Expression)) for oo in [dup_obj, obj])

                # Get list of duplicated objects or an empy list
                dup_objects = self.duplicated_expressions[obj.name]
                if len(dup_objects) == 0:
                    dup_objects.append(dup_obj)
                dup_objects.append(obj)

        # Update global information about ode object
        self.present_ode_objects[obj.name] = PresentObjTuple(obj, comp)
        self.ns.update({obj.name : obj.sym})

        # If Expression
        if isinstance(obj, Expression):

            # Append the name to the list of all ordered components with
            # expressions. If the ODE is finalized we do not update components
            if not self._is_finalized_ode:
                self._handle_expr_component(comp, obj)

            # Add dependencies between registered comments and expressions so
            # they are carried over in Code components
            for comment in comp._local_comments:
                self.object_used_in[comment].add(obj)
                self.expression_dependencies[obj].add(comment)

            # Expand and add any derivatives in the expressions
            expression_added = False
            for der_expr in obj.expr.atoms(sp.Derivative):
                expression_added |= self._expand_single_derivative(comp, obj, der_expr)

            # If any expression was added we need to bump the count of the ODEObject
            if expression_added:
                obj._recount()

            # Expand the Expression
            self.expanded_expressions[obj.name] = self._expand_expression(obj)

            # If the expression is a StateSolution the state cannot have
            # been used previously
            if isinstance(obj, StateSolution) and \
                   self.object_used_in.get(obj.state):
                used_in = self.object_used_in.get(obj.state)
                error("A state solution cannot have been used in "\
                      "any previous expressions. {0} is used in: {1}".format(\
                          obj.state, used_in))

    def _handle_expr_component(self, comp, expr):
        """
        A help function to sort and add components in the ordered
        the intermediate expressions are added to the ODE
        """
        
        if len(self.all_expr_components_ordered) == 0:
            self.all_expr_components_ordered.append(comp.name)

            # Add a comment to the component
            comp.add_comment("Intermediate expressions for the {0} "\
                             "component".format(comp.name))

            # Recount the last added expression so the comment comes
            # infront of the expression
            expr._recount()

        # We are shifting expression components
        elif self.all_expr_components_ordered[-1] != comp.name:

            # Finalize the last component we visited
            self.all_components[\
                self.all_expr_components_ordered[-1]].finalize_component()
                
            # Append this component
            self.all_expr_components_ordered.append(comp.name)

            # Add a comment to the component
            comp.add_comment("Intermediate expressions for the {0} "\
                             "component".format(comp.name))

            # Recount the last added expression so the comment comes
            # infront of the expression
            expr._recount()
        
    def _expand_single_derivative(self, comp, obj, der_expr):
        """
        Expand a single derivative and register it as new derivative expression
        
        Returns True if an expression was actually added
        """

        if not isinstance(der_expr.args[0], AppliedUndef):
            error("Can only register Derivatives of allready registered "\
            "Expressions. Got: {0}".format(sympycode(der_expr.args[0])))

        if not isinstance(der_expr.args[1], (AppliedUndef, sp.Symbol)):
            error("Can only register Derivatives with a single dependent "\
                  "variabe. Got: {0}".format(sympycode(der_expr.args[1])))

        # Try accessing already registered derivative expressions
        der_expr_obj = self.present_ode_objects.get(sympycode(der_expr))

        # If excist continue
        if der_expr_obj:
            return False

        # Get the expr and dependent variable objects
        expr_obj = self.present_ode_objects[sympycode(der_expr.args[0])].obj
        var_obj = self.present_ode_objects[sympycode(der_expr.args[1])].obj

        # If the dependent variable is time and the expression is a state
        # variable we raise an error as the user should already have created
        # the expression.
        if isinstance(expr_obj, State) and var_obj == self._time:
            error("The expression {0} is dependent on the state "\
                  "derivative of {1} which is not registered in this ODE."\
                  .format(obj, expr_obj))

        if not isinstance(expr_obj, Expression):
            error("Can only differentiate expressions or states. Got {0} as "\
                  "the derivative expression.".format(expr_obj))

        # If we get a Derivative(expr, t) we issue an error
        if isinstance(expr_obj, Expression) and var_obj == self._time:
            error("All derivative expressions of registered expressions "\
                  "need to be expanded with respect to time. Use "\
                  "expr.diff(t) instead of Derivative(expr, t) ")

        # Store expression
        comp.add_derivative(expr_obj, var_obj, expr_obj.expr.diff(var_obj.sym))

        return True

    def _expand_expression(self, obj):

        timer = Timer("Expanding expression")

        # FIXME: We need to wait for the expanssion of all expressions...
        assert isinstance(obj, Expression)

        # Iterate over dependencies in the expression
        expression_subs_dict = {}
        for sym in symbols_from_expr(obj.expr, include_derivatives=True):

            dep_obj = self.present_ode_objects[sympycode(sym)]

            if dep_obj is None:
                error("The symbol '{0}' is not declared within the '{1}' "\
                      "ODE.".format(sym, self.name))

            # Expand dep_obj
            dep_obj, dep_comp = dep_obj.obj, dep_obj.comp

            # Store object dependencies
            self.expression_dependencies[obj].add(dep_obj)
            self.object_used_in[dep_obj].add(obj)

            # Expand dependent expressions
            if isinstance(dep_obj, Expression):

                # Collect intermediates to be used in substitutions below
                expression_subs_dict[dep_obj.sym] = self.expanded_expressions[dep_obj.name]

        return obj.expr.xreplace(expression_subs_dict)

    @property
    def mass_matrix(self):
        """
        Return the mass matrix as a sympy.Matrix
        """

        if not self.is_finalized:
            error("The ODE must be finalized")
            
        if not self._mass_matrix:
        
            state_exprs = self.state_expressions
            N = len(state_exprs)
            self._mass_matrix = sp.Matrix(N, N, lambda i, j : 1 if i==j and \
                        isinstance(state_exprs[i], StateDerivative) else 0)
            
        return self._mass_matrix

    @property
    def is_dae(self):
        """
        Return True if ODE is a DAE
        """
        if not self.is_complete:
            error("The ODE is not complete")

        return any(isinstance(expr, AlgebraicExpression) for expr in \
                   self.state_expressions)

    def finalize(self):
        """
        Finalize the ODE
        """
        for comp in self.components:
            comp.finalize_component()
            
        self._is_finalized_ode = True
        self._present_component = self.name

    def signature(self):
        """
        Return a signature uniquely defining the ODE
        """
        import hashlib
        def_list = [repr(state.param) for state in self.full_states] 
        def_list += [repr(param.param) for param in self.parameters] 
        def_list += [str(expr.expr) for expr in self.intermediates]
        def_list += [str(expr.expr) for expr in self.state_expressions]

        h = hashlib.sha1()
        h.update(";".join(def_list))
        return h.hexdigest()
