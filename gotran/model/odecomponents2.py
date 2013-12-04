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

__all__ = ["ODE", "ODEObjectList", "ODEBaseComponent"]

# System imports
from collections import OrderedDict, defaultdict, deque
import re
import types

from sympy.core.function import AppliedUndef

# ModelParameters imports
from modelparameters.sympytools import sp, symbols_from_expr
from modelparameters.parameters import ScalarParam
from modelparameters.parameterdict import ParameterDict
from modelparameters.codegeneration import sympycode, _all_keywords
from modelparameters.utils import tuplewrap, Timer

# Local imports
from gotran.common import error, debug, check_arg, check_kwarg, scalars
from gotran.model.odeobjects2 import *
from gotran.model.expressions2 import *

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

class iter_objects(object):
    """
    A recursive iterator over all objects of a component including its childrens

    Arguments
    ---------
    comp : ODEBaseComponent
        The root ODEComponent of the iteration
    reverse : bool
        If True the iteration is done from the last component added
    types : ODEObject types (optional)
        Only iterate over particular types

    Yields
    ------
    ode_object : ODEObject
        All ODEObjects of a component
    """
    def __init__(self, comp, return_comp=True, only_return_comp=False,
                 reverse=False, *types):
        assert isinstance(comp, ODEBaseComponent)
        self._types = tuplewrap(types) or (ODEObject,)
        self._return_comp = return_comp
        self._only_return_comp = only_return_comp
        if reverse:
            self._object_iterator = self._reverse_iter_objects(comp)
        else:
            self._object_iterator = self._iter_objects(comp)

        assert all(issubclass(T, ODEObject) for T in self._types)

    def _reverse_iter_objects(self, comp):

        # First all children components in reversed order
        for sub_comp in reversed(comp.children.values()):
            for sub_tree in self._reverse_iter_objects(sub_comp):
                yield sub_tree

        # Secondly return component
        if self._return_comp or self._only_return_comp:
            yield comp

        if self._only_return_comp:
            return

        # Last all objects
        for obj in reversed(comp.ode_objects):
            if isinstance(obj, self._types):
                yield obj

    def _iter_objects(self, comp):

        # First return component
        if self._return_comp:
            yield comp

        # Secondly all objects
        if not self._only_return_comp:

            for obj in comp.ode_objects:
                if isinstance(obj, self._types):
                    yield obj

        # Thrirdly all children components
        for sub_comp in comp.children.values():
            for sub_tree in self._iter_objects(sub_comp):
                yield sub_tree

    def next(self):
        return self._object_iterator.next()

    def __iter__(self):
        return self

def ode_objects(comp, *types):
    """
    Return a list of ode objects

    Arguments
    ---------
    comp : ODEBaseComponent
        The root ODEComponent of the list
    types : ODEObject types (optional)
        Only include objects of type given in types
    """
    return [obj for obj in iter_objects(comp, False, False, False, *types)]

def ode_components(comp, include_self=True):
    """
    Return a list of ode components

    Arguments
    ---------
    comp : ODEBaseComponent
        The root ODEComponent of the list
    return_self : bool (optional)
        The list will include the passed component if True
    """
    comps = [obj for obj in iter_objects(comp, True, True)]
    if not include_self:
        comps.remove(comp)

    return comps

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
    if state_comp and isinstance(state_comp[0], State):
        return state_comp[0], STATE_SOLUTION_EXPRESSION

    return None, INTERMEDIATE

class ODEBaseComponent(ODEObject):
    """
    Base class for all ODE components. 
    """
    def __init__(self, name, parent):
        """
        Create an ODEBaseComponent

        Arguments
        ---------
        name : str
            The name of the component. This str serves as the unique
            identifier of the Component.
        parent : ODEBaseComponent
            The parent component of this ODEComponent
        """

        self._constructed = False
        check_arg(name, str, 0, ODEBaseComponent)
        check_arg(parent, ODEBaseComponent, 1, ODEBaseComponent)

        # Call super class
        super(ODEBaseComponent, self).__init__(name)

        # Store parent component
        self.parent = parent

        # Store ODEBaseComponent children
        self.children = OrderedDict()

        # Store ODEObjects of this component
        self.ode_objects = ODEObjectList()

        # Store all state expressions
        self._local_state_expressions = dict()

        # Flag to check if component is finalized
        self._is_finalized = False

        self._constructed = True

    def get_object(self, name, reversed=True, return_component=False):

        # First check self
        obj = self.ode_objects.get(name)

        if obj:
            return self, obj

        # Then iterate all objects in ode
        comp, obj = None, None

        # If a name is registered
        if name in self.root.ns:

            for obj in iter_objects(self, True, False, reversed):
                if isinstance(obj, ODEBaseComponent):
                    comp = obj
                elif obj.name == name:
                    break
            else:
                comp, obj = None, None

        return comp, obj if return_component else obj

    @property
    def t(self):
        """
        Return the time symbol
        """
        return self.root._time.sym

    @property
    def time(self):
        """
        Return the time
        """
        return self.root._time

    def add_comment(self, comment):
        """
        Add a comment to the ODE component

        Arguments
        ---------
        comment : str
            The comment
        """
        self.ode_objects.append(Comment(comment))

    def add_state(self, name, init):
        """
        Add a state to the component

        Arguments
        ---------
        name : str
            The name of the state variable
        init : scalar, ScalarParam
            The initial value of the state
        """
        timer = Timer("Add states")

        # Create state
        state = State(name, init, self.time)

        self._register_component_object(state)

        # Return the sympy version of the state
        return state.sym

    def add_states(self, *args, **kwargs):
        """
        Add a number of states to the current ODEComponent

        Arguments
        ---------
        args : list of tuples
            A list of tuples with states and init values. Use this to
            set states if you need them ordered.
        kwargs : dict
            A dict with states
        """

        states = list(args) + sorted(kwargs.items())

        if len(states) == 0:
            error("Expected at least one state")

        for arg in states:
            if not isinstance(arg, tuple) or len(arg) != 2:
                error("excpected tuple with lenght 2 with state name (str) "\
                      "and init values as the args argument.")
            state_name, init = arg

            # Add the states
            self.add_state(state_name, init)

    def add_parameter(self, name, init):
        """
        Add a parameter to the component

        Arguments
        ---------
        name : str
            The name of the parameter
        init : scalar, ScalarParam
            The initial value of the parameter
        """
        timer = Timer("Add parameters")

        param = Parameter(name, init)

        self._register_component_object(param)

        # Return the sympy version of the state
        return param.sym

    def add_parameters(self, *args, **kwargs):
        """
        Add a number of parameters to the current ODEComponent

        Arguments
        ---------
        args : list of tuples
            A list of tuples with parameters and init values. Use this to
            set parameters if you need them ordered.
        kwargs : dict
            A dict with parameters
        """

        params = list(args) + sorted(kwargs.items())

        if len(params) == 0:
            error("expected at least one parameter")

        for arg in params:
            if not isinstance(arg, tuple) or len(arg) != 2:
                error("excpected tuple with lenght 2 with parameter name (str) "\
                      "and init values as the args argument.")
            parameter_name, value = arg

            # Add the parameters
            self.add_parameter(parameter_name, value)

    def add_component(self, name):
        """
        Add a sub ODEComponent
        """
        comp = DerivativeComponent(name, self)

        if name in self.root.all_components:
            error("A component with the name '{0}' already excists.".format(name))

        self.children[comp.name] = comp
        self.root.all_components[comp.name] = comp
        self.root._present_component = comp.name

        return comp

    def add_markov_model(self, name):
        """
        Add a sub MarkovModelComponent
        """
        comp = MarkovModelComponent(name, self)

        if name in self.root.all_components:
            error("A component with the name '{0}' already excists.".format(name))

        self.children[comp.name] = comp
        self.root.all_components[comp.name] = comp
        self.root._present_component = comp.name

        return comp

    def add_solve_state(self, state, expr, **solve_flags):
        """
        Add a solve state expression which tries to find a solution to
        a state by solving an algebraic expression

        Arguments
        ---------
        state : State, AppliedUndef
            The State that is solved
        expr : sympy.Basic
            The expression that determines the state
        solve_flags : dict
            Flags that are passed directly to sympy.solve
        """

        state = self._expect_state(state)

        # Check the sympy flags
        if solve_flags.get("dict") or solve_flags.get("set"):
            error("Cannot use dict=True or set=True as sympy_flags")

        # Check that there are no expressions that are dependent on the state
        for sym in symbols_from_expr(expr, include_derivatives=True):
            if (state.sym not in sym) and (state.sym in sym):
                error("{0}, a sub expression of the expression, cannot depend "\
                      "on the state for which we try to solve for.".format(sym))

        # Try solve the passed expr
        try:
            solved_expr = sp.solve(expr, state.sym)
        except:
            error("Could not solve the passed expression")

        assert isinstance(solved_expr, list)

        # FIXME: Add possibilities to choose solution?
        if len(solved_expr) != 1:
            error("expected only 1 solution")

        # Unpack the solution
        solved_expr = solved_expr[0]

        self.add_state_solution(state, solved_expr)

    def add_state_solution(self, state, expr):
        """
        Add a solution expression for a state
        """

        state = self._expect_state(state)

        if "d{0}_dt".format(state.name) in self.ode_objects:
            error("Cannot registered a state solution for a state "\
                  "that has a state derivative registered.")

        if "alg_{0}_0".format(state.name) in self.ode_objects:
            error("Cannot registered a state solution for a state "\
                  "that has an algebraic expression registered.")

        # Create a StateSolution in the present component
        obj = StateSolution(state, expr)

        self._register_component_object(obj)

    def add_intermediate(self, name, expr):
        """
        Register an intermediate math expression

        Arguments
        ---------
        name : str
            The name of the expression
        expr : sympy.Basic, scalar
            The expression
        """

        # Create an Intermediate in the present component
        timer = Timer("Add intermediate")
        expr = Intermediate(name, expr)

        self._register_component_object(expr)

        return expr.sym

    def add_derivative(self, der_expr, dep_var, expr):
        """
        Add a derivative expression

        Arguments
        ---------
        der_expr : Expression, State, sympy.AppliedUndef
            The Expression or State which is differentiated
        dep_var : State, Time, Expression, sympy.AppliedUndef, sympy.Symbol
            The dependent variable
        expr : sympy.Basic
            The expression which the differetiation should be equal
        """
        timer = Timer("Add derivatives")

        if isinstance(der_expr, AppliedUndef):
            name = sympycode(der_expr)
            der_expr = self.root.present_ode_objects.get(name)

            if der_expr is None:
                error("{0} is not registered in this ODE".format(name))
            der_expr = der_expr[0]

        if isinstance(dep_var, (AppliedUndef, sp.Symbol)):
            name = sympycode(dep_var)
            dep_var = self.root.present_ode_objects.get(name)

            if dep_var is None:
                error("{0} is not registered in this ODE".format(name))
            dep_var = dep_var[0]

        # Check if der_expr is a State
        if isinstance(der_expr, State):
            self._expect_state(der_expr)
            obj = StateDerivative(der_expr, expr)

        else:

            # Create a DerivativeExpression in the present component
            obj = DerivativeExpression(der_expr, dep_var, expr)

        self._register_component_object(obj)

        return obj.sym

    def add_algebraic(self, state, expr):
        """
        Add an algebraic expression which relates a State with an
        expression which should equal to 0

        Arguments
        ---------
        state : State
            The State which the algebraic expression should determine
        expr : sympy.Basic
            The expression that should equal 0
        """

        state = self._expect_state(state)

        if "d{0}_dt".format(state.name) in self.ode_objects:
            error("Cannot registered an algebraic expression for a state "\
                  "that has a state derivative registered.")

        if state.is_solved:
            error("Cannot registered an algebraic expression for a state "\
                  "which is registered solved.")

        # Create an AlgebraicExpression in the present component
        obj = AlgebraicExpression(state, expr)

        self._register_component_object(obj)

    @property
    def states(self):
        """
        Return a list of all states in the component and its children
        """
        return [state for state in iter_objects(self, False, False, False, \
                                                State)]

    @property
    def full_states(self):
        """
        Return a list of all states in the component and its children that are
        not solved and determined by a state expression
        """
        return [obj for obj in iter_objects(self, False, False, False, \
                                            State) if not obj.is_solved]

    @property
    def full_state_vector(self):
        """
        Return a sympy column vector with all full states
        """
        states = self.full_states
        
        return sp.Matrix(len(states), 1, lambda i, j: states[i].sym)

    @property
    def field_states(self):
        """
        Return a list of all field states in the component
        """
        return [obj for obj in iter_objects(self, False, False, False, State)\
                if obj.is_field]

    @property
    def all_parameters(self):
        """
        Return a list of all parameters in the component
        """
        return ode_objects(self, Parameter)

    @property
    def parameters(self):
        """
        Return a list of all parameters in the component which are not field
        parameters
        """
        return [obj for obj in iter_objects(self, False, False, False, \
                                            Parameter) if not obj.is_field]

    @property
    def field_parameters(self):
        """
        Return a list of all field parameters in the component
        """
        return [obj for obj in iter_objects(self, False, False, False, \
                                            Parameter) if obj.is_field]

    @property
    def intermediates(self):
        """
        Return a list of all intermediates
        """
        return [obj for obj in iter_objects(self, False, False, False, \
                                            Intermediate)]

    @property
    def state_expressions(self):
        """
        Return a list of state expressions
        """
        states = self.full_states
        return sorted((obj for obj in iter_objects(self, False, False, \
                                                   False, StateExpression)),\
                      lambda o0, o1 : cmp(states.index(o0.state), \
                                          states.index(o1.state)))

    @property
    def components(self):
        """
        Return a list of all child components in the component
        """
        return ode_components(self)

    @property
    def root(self):
        """
        Return the root ODE component (the ode)
        """
        present = self
        while present != present.parent:
            present = present.parent

        return present

    @property
    def is_finalized(self):
        return self._is_finalized
    
    @property
    def num_states(self):
        """
        Return the number of all states
        """
        return len(self.states)

    @property
    def num_full_states(self):
        """
        Return the number of all full states
        """
        return len(self.full_states)

    @property
    def num_field_states(self):
        """
        Return the number of all field states
        """
        return len(self.field_states)

    @property
    def num_all_parameters(self):
        """
        Return the number of all parameters
        """
        return len(self.all_parameters)

    @property
    def num_parameters(self):
        """
        Return the number of all parameters
        """
        return len(self.parameters)

    @property
    def num_field_parameters(self):
        """
        Return the number of all field parameters
        """
        return len(self.field_parameters)

    @property
    def num_intermediates(self):
        """
        Return the number of all intermediates
        """
        return len(self.intermediates)

    @property
    def num_state_expressions(self):
        """
        Return the number state expressions
        """
        return len(self.state_expressions)

    @property
    def num_components(self):
        """
        Return the number of all components including it self
        """
        return len(self.components)

    @property
    def is_complete(self):
        """
        True if the component and all its children are locally complete
        """
        return self.is_locally_complete and all(child.is_complete for child \
                                                in self.children.values())

    @property
    def is_locally_complete(self):
        """
        True if the number of non-solved states are the same as the number
        of registered state expressions
        """
        num_local_states = sum(1 for obj in self.ode_objects \
                               if isinstance(obj, State) and not obj.is_solved)

        return num_local_states == len(self._local_state_expressions) 

    def __call__(self, name):
        """
        Return a child component, if the component does not excist, create
        and add one
        """
        check_arg(name, str)

        comp = self.children.get(name)
        if comp is None:
            comp = self.add_component(name)
            debug("Adding '{0}' component to {1}".format(name, self))
        else:
            self.root._present_component = comp.name
            
        return comp

    def _expect_state(self, state, allow_state_solution=False):
        """
        Help function to check an argument which should be expected
        to be a state
        """

        if allow_state_solution:
            allowed = (State, StateSolution)
        else:
            allowed = (State,)

        if isinstance(state, AppliedUndef):
            name = sympycode(state)
            state_comp = self.root.present_ode_objects.get(name)

            if state_comp is None:
                error("{0} is not registered in this ODE".format(name))

            state, comp = state_comp

        check_arg(state, allowed, 0)

        if isinstance(state, State) and  state.is_solved:
            error("Cannot registered a state expression for a state "\
                  "which is registered solved.")

        return state

    def _register_component_object(self, obj):
        """
        Register an ODEObject to the component
        """
        
        if self._is_finalized:
            error("Cannot add {0} {1} to component {2} it is "\
                  "already finalized.".format(\
                      obj.__class__.__name__, obj, self))
        
        self._check_reserved_wordings(obj)

        # If registering a StateExpression
        if isinstance(obj, StateExpression):
            if obj.state in self._local_state_expressions:
                error("A StateExpression for state {0} is already registered "\
                      "in this component.")

            # Check that the state is registered in this component
            state_obj = self.ode_objects.get(obj.state.name)
            if not isinstance(state_obj, State):
                error("The state expression {0} defines state {1}, which is "\
                      "not registered in the {2} component.".format(\
                          obj, obj.state, self))

            self._local_state_expressions[obj.state] = obj

        # If obj is Intermediate register it as an attribute so it can be used
        # later on.
        # FIXME: This should pretty much be always true...
        if isinstance(obj, (State, Parameter, Expression)):

            # Register symbol, overwrite any already excisting symbol
            self.__dict__[obj.name] = obj.sym

        # Register the object in the root ODE,
        # (here all duplication checks and expression expansions are done)
        self.root.register_ode_object(obj, self)

        # Register the object
        self.ode_objects.append(obj)

    def _check_reserved_wordings(self, obj):
        if obj.name in _all_keywords:
            error("Cannot register a {0} with a computer language "\
                  "keyword name: {1}".format(obj.__class__.__name__,
                                             obj.name))

        # Check for reserved Expression wordings
        #if isinstance(obj, Expression):
        #    if re.search(_derivative_name_template, obj.name) \
        #           and not isinstance(obj, Derivatives):
        #        error("The pattern d{{name}}_dt is reserved for derivatives. "
        #              "However {0} is not a Derivative.".format(obj.name))
        #    
        #    if re.search(_algebraic_name_template, obj.name) \
        #           and not isinstance(obj, AlgebraicExpression):
        #        error("The pattern {alg_{{name}}_0 is reserved for algebraic "\
        #              "expressions, however {1} is not an AlgebraicExpression."\
        #              .format(obj.name))

    def finalize_component(self):
        """
        Called whenever the component should be finalized
        """
        if self._is_finalized:
            return

        if not self.is_locally_complete:
            error("Cannot finalize component '{0}'. It is "\
                  "not complete.".format(self))
            
        #if self.is_finalized:
        #    error("Cannot finalize component '{0}'. It is already "\
        #          "finalized.".format(self))

        self._is_finalized = True

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
                self.add_derivative(expr_obj[0], var_obj[0], value)

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

    def __init__(self, name, ns=None):

        # Call super class with itself as parent component
        super(ODE, self).__init__(name, self)

        # Reset constructed attribute
        self._constructed = False

        self.ns = ns if ns is not None else {}

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
        self.all_components = {name : self}

        # An attribute keeping track of the present ODE component
        self._present_component = self.name

        # Keep track of monitored intermediates
        self.monitored_intermediates = OrderedDict()

        # A dict with the present ode objects
        # NOTE: hashed by name so duplicated expressions are not stored
        self.present_ode_objects = dict(t=(self._time, self), time=(self._time, self))

        # Keep track of duplicated expressions
        self.duplicated_expressions = defaultdict(list)

        # Keep track of expression dependencies and in what expression
        # an object has been used in
        self.expression_dependencies = defaultdict(list)
        self.object_used_in = defaultdict(list)

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

            dup_obj, dup_comp = duplication

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
        self.present_ode_objects[obj.name] = (obj, comp)
        self.ns.update({obj.name : obj.sym})

        # If Expression
        if isinstance(obj, Expression):

            # Append the name to the list of all ordered components with
            # expressions. If the ODE is finalized we do not update components
            if not self._is_finalized_ode:
                self._handle_expr_component(comp)

            # Expand and add any derivatives in the expressions
            for der_expr in obj.expr.atoms(sp.Derivative):
                self._expand_single_derivative(comp, obj, der_expr)

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

    def _handle_expr_component(self, comp):
        """
        A help function to sort and add components in the ordered
        the intermediate expressions are added to the ODE
        """
        
        if len(self.all_expr_components_ordered) == 0:
            self.all_expr_components_ordered.append(comp.name)

            # Add a comment to the component
            comp.add_comment("Intermediate expressions for the "\
                             "{0} component".format(comp.name))
        
        # We are shifting expression components
        elif self.all_expr_components_ordered[-1] != comp.name:

            # Finalize the last component we visited
            self.all_components[\
                self.all_expr_components_ordered[-1]].finalize_component()
                
            # Append this component
            self.all_expr_components_ordered.append(comp.name)

            # Add a comment to the component
            comp.add_comment("Intermediate expressions for the "\
                                 "{0} component".format(comp.name))

    def _expand_single_derivative(self, comp, obj, der_expr):
        """
        Expand a single derivative and register it as new derivative expression
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
            return

        # Get the expr and dependent variable objects
        expr_obj = self.present_ode_objects[sympycode(der_expr.args[0])][0]
        var_obj = self.present_ode_objects[sympycode(der_expr.args[1])][0]

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
            dep_obj, dep_comp = dep_obj

            # Store object dependencies
            self.expression_dependencies[obj].append(dep_obj)
            self.object_used_in[dep_obj].append(obj)

            # Expand depentent expressions
            if isinstance(dep_obj, Expression):

                # Collect intermediates to be used in substitutions below
                expression_subs_dict[dep_obj.sym] = self.expanded_expressions[dep_obj.name]

        return obj.expr.xreplace(expression_subs_dict)

    @property
    def body_expressions(self):
        """
        Return a list of all body expressions if ODE is finalized it will be cached
        """
        
        if self._body_expressions:
            return self._body_expressions

        body_expressions = []

        # Iterate over all components
        for comp_name in self.all_expr_components_ordered:

            comp = self.all_components[comp_name]

            # Iterate over all objects of the component
            for obj in comp.ode_objects:

                # Only add Expressions
                if isinstance(obj, Expression):
                    body_expressions.append(obj)

        if self.is_finalized:
            self._body_expressions = body_expressions

        return body_expressions

    def add_monitored(self, *args):
        """
        Add intermediate expressions to be monitored

        Arguments
        ---------
        args : any number of intermediates
            Intermediates which will be monitored
        """

        for i, arg in enumerate(args):
            check_arg(arg, (str, sp.Symbol, AppliedUndef), i)
            name = arg if isinstance(str) else sympycode(arg)
            obj = self.present_ode_objects.get(name)

            if not isinstance(obj, Expression):
                error("Can only monitor Expressions. '{0}' is not an "\
                      "Expression.".format(obj.name))
            
            # Register the expanded monitored intermediate
            self.monitored_intermediates[obj.name] = obj

    @property
    def num_monitored_intermediates(self):
        return len(self.monitored_intermediates)

    def mass_matrix(self):
        """
        Return the mass matrix as a sympy.Matrix
        """

        if not self.is_complete:
            error("The ODE is not complete")
            
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
        self._compute_argument_indices()

    def _compute_argument_indices(self):
        """
        Compute the argument indices for this ODE (States, StateExpressions, Parameters)
        """
        self._arg_indices = {}
        
        for ind, state_expr in enumerate(self.state_expressions):
            self._arg_indices[state_expr] = ind
            self._arg_indices[state_expr.state] = ind

        for ind, param in enumerate(self.parameters):
            self._arg_indices[param] = ind
        
        for ind, param in enumerate(self.field_parameters):
            self._arg_indices[param] = ind

        ind = 0
        for expr in self.body_expressions:
            if isinstance(expr, Comment):
                continue
            self._arg_indices[expr] = ind
            ind += 1
        
    def arg_index(self, arg):
        """
        Return the argument index of an ODEValueObject
        """
        check_arg(arg, ODEValueObject, 0, ODE.arg_index)

        if not self.is_finalized:
            error("ODE must be finalized to extract argument indices")
        index = self._arg_indices.get(arg)
        if index is None:
            error("No index registered for {0}".format(arg))
        return index

    def arg_cmp(self, arg0, arg1):
        """
        Compare method to sort a list of arguments
        """
        return cmp(self.arg_index(arg0), self.arg_index(arg1))

# Is it nesseary?
class CodeComponent(object):
    pass

class ODEObjectList(list):
    """
    Specialized container for ODEObjects
    """
    def __init__(self):
        """
        Initialize ODEObjectList. Only empty such.
        """
        super(ODEObjectList, self).__init__()
        self._objects = {}

    def keys(self):
        return self._objects.keys()

    def append(self, item):
        check_arg(item, ODEObject, 0, ODEObjectList.append)
        super(ODEObjectList, self).append(item)
        self._objects[item.name] = item

    def insert(self, index, item):
        check_arg(item, ODEObject, 1, ODEObjectList.insert)
        super(ODEObjectList, self).insert(index, item)
        self._objects[item.name] = item

    def extend(self, iterable):
        check_arg(iterable, list, 0, ODEObjectList.extend, ODEObject)
        super(ODEObjectList, self).extend(iterable)
        for item in iterable:
            self._objects[item.name] = item

    def get(self, name):
        if isinstance(name, str):
            return self._objects.get(name)
        elif isinstance(name, sp.Symbol):
            return self._objects.get(name.name)
        return None

    def __contains__(self, item):
        if isinstance(item, str):
            return any(item == obj.name for obj in self)
        elif isinstance(item, sp.Symbol):
            return any(item.name == obj.name for obj in self)
        elif (item, ODEObject):
            return super(ODEObjectList, self).__contains__(item)
        return False

    def count(self, item):
        if isinstance(item, str):
            return sum(item == obj.name for obj in self)
        elif isinstance(item, sp.Symbol):
            return sum(item.name == obj.name for obj in self)
        elif (item, ODEObject):
            return super(ODEObjectList, self).count(item)
        return 0

    def index(self, item):
        if isinstance(item, str):
            for ind, obj in enumerate(self):
                if item == obj.name:
                    return ind
        elif isinstance(item, sp.Symbol):
            for ind, obj in enumerate(self):
                if item.name == obj.name:
                    return ind
        elif (item, ODEObject):
            for ind, obj in enumerate(self):
                if item == obj:
                    return ind
        raise ValueError("Item '{0}' not part of this ODEObjectList.".format(str(item)))

    def sort(self):
        error("Cannot sort ODEObjectList.")

    def pop(self, index):

        check_arg(index, int)
        if index >= len(self):
            raise IndexError("pop index out of range")
        obj=super(ODEObjectList, self).pop(index)
        self._objects.pop(obj.name)

    def remove(self, item):
        try:
            index = self.index(item)
        except ValueError:
            raise ValueError("ODEObjectList.remove(x): x not in list")

        self.pop(index)

    def reverse(self, item):
        error("Cannot alter ODEObjectList, other than adding ODEObjects.")


