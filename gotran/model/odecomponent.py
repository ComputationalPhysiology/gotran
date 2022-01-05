# Copyright (C) 2013-2014 Johan Hake
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

__all__ = ["ODEComponent"]

import weakref

# System imports
from collections import OrderedDict, defaultdict

from modelparameters.codegeneration import _all_keywords, sympycode

# Local imports
from modelparameters.logger import debug, error

# ModelParameters imports
from modelparameters.sympytools import sp, symbols_from_expr
from modelparameters.utils import Timer, check_arg, scalars
from modelparameters.sympy.core.function import AppliedUndef

from .expressions import (
    AlgebraicExpression,
    DerivativeExpression,
    Expression,
    IndexedExpression,
    IndexedObject,
    Intermediate,
    RateExpression,
    State,
    StateDerivative,
    StateExpression,
    StateSolution,
)
from .odeobjects import Comment, ODEObject, Parameter
from .utils import (
    ALGEBRAIC_EXPRESSION,
    DERIVATIVE_EXPRESSION,
    INTERMEDIATE,
    STATE_SOLUTION_EXPRESSION,
    ODEObjectList,
    RateDict,
    iter_objects,
    ode_components,
    ode_objects,
    special_expression,
)


class ODEComponent(ODEObject):
    """
    Base class for all ODE components.
    """

    def __init__(self, name, parent):
        """
        Create an ODEComponent

        Arguments
        ---------
        name : str
            The name of the component. This str serves as the unique
            identifier of the Component.
        parent : gotran.ODEComponent
            The parent component of this ODEComponent
        """

        # Turn off magic attributes (see __setattr__ method) during
        # construction
        self._allow_magic_attributes = False

        check_arg(name, str, 0, ODEComponent)
        check_arg(parent, ODEComponent, 1, ODEComponent)

        # Call super class
        super(ODEComponent, self).__init__(name)

        # Store parent component
        self._parent = weakref.ref(parent)

        # Store ODEComponent children
        self.children = OrderedDict()

        # Store ODEObjects of this component
        self.ode_objects = ODEObjectList()

        # Storage of rates
        self.rates = RateDict(self)

        # Store all state expressions
        self._local_state_expressions = dict()

        # Collect all comments
        self._local_comments = []

        # Flag to check if component is finalized
        self._is_finalized = False
        self._is_finalizing = False

        # Turn on magic attributes (see __setattr__ method)
        self._allow_magic_attributes = True

    @property
    def parent(self):
        """
        Return the the parent if it is still alive. Otherwise it will return None
        """
        p = self._parent()
        if p is None:
            error("Cannot access parent ODEComponent. It is already " "destroyed.")
        return p

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
                if isinstance(obj, ODEComponent):
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

    def add_comment(self, comment, dependent=None):
        """
        Add a comment to the ODE component

        Arguments
        ---------
        comment : str
            The comment
        dependent : gotran.ODEObject
            If given the count of this comment will follow as a
            fractional count based on the count of the dependent object
        """
        comment = Comment(comment, dependent)
        self.ode_objects.append(comment)
        self._local_comments.append(comment)

    def add_state(self, name, init):
        """
        Add a state to the component

        Arguments
        ---------
        name : str
            The name of the state variable
        init : scalar, modelparameters.ScalarParam
            The initial value of the state
        """
        timer = Timer("Add states")  # noqa: F841

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
                error(
                    "excpected tuple with lenght 2 with state name (str) "
                    "and init values as the args argument.",
                )
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
        init : scalar, modelparameters.ScalarParam
            The initial value of the parameter
        """
        timer = Timer("Add parameters")  # noqa: F841

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
                error(
                    "excpected tuple with lenght 2 with parameter name (str) "
                    "and init values as the args argument.",
                )
            parameter_name, value = arg

            # Add the parameters
            self.add_parameter(parameter_name, value)

    def add_component(self, name):
        """
        Add a sub ODEComponent
        """

        comp = ODEComponent(name, self)

        if name in self.root.all_components:
            error(f"A component with the name '{name}' already excists.")

        self.children[comp.name] = comp
        self.root.all_components[comp.name] = comp
        self.root._present_component = comp.name

        return comp

    def add_solve_state(self, state, expr, dependent=None, **solve_flags):
        """
        Add a solve state expression which tries to find a solution to
        a state by solving an algebraic expression

        Arguments
        ---------
        state : gotran.State, AppliedUndef
            The State that is solved
        expr : sympy.Basic
            The expression that determines the state
        dependent : gotran.ODEObject
            If given the count of this expression will follow as a
            fractional count based on the count of the dependent object
        solve_flags : dict
            Flags that are passed directly to sympy.solve
        """

        from modelparameters.sympytools import symbols_from_expr

        state = self._expect_state(state)

        # Check the sympy flags
        if solve_flags.get("dict") or solve_flags.get("set"):
            error("Cannot use dict=True or set=True as sympy_flags")

        # Check that there are no expressions that are dependent on the state
        for sym in symbols_from_expr(expr, include_derivatives=True):
            if (state.sym is not sym) and (state.sym in sym.atoms()):
                error(
                    "{0}, a sub expression of the expression, cannot depend "
                    "on the state for which we try to solve for.".format(sym),
                )

        # Try solve the passed expr
        try:
            solved_expr = sp.solve(expr, state.sym)
        except Exception:
            error("Could not solve the passed expression")

        assert isinstance(solved_expr, list)

        # FIXME: Add possibilities to choose solution?
        if len(solved_expr) != 1:
            error("expected only 1 solution")

        # Unpack the solution
        solved_expr = solved_expr[0]

        self.add_state_solution(state, solved_expr, dependent)

    def add_state_solution(self, state, expr, dependent=None):
        """
        Add a solution expression for a state

        Arguments
        ---------
        state : gotran.State, AppliedUndef
            The State that is solved
        expr : sympy.Basic
            The expression that determines the state
        dependent : gotran.ODEObject
            If given the count of this expression will follow as a
            fractional count based on the count of the dependent object
        """

        state = self._expect_state(state)

        if f"d{state.name}_dt" in self.ode_objects:
            error(
                "Cannot registered a state solution for a state "
                "that has a state derivative registered.",
            )

        if f"alg_{state.name}_0" in self.ode_objects:
            error(
                "Cannot registered a state solution for a state "
                "that has an algebraic expression registered.",
            )

        # Create a StateSolution in the present component
        obj = StateSolution(state, expr, dependent)

        self._register_component_object(obj)

    def add_intermediate(self, name, expr, dependent=None):
        """
        Register an intermediate math expression

        Arguments
        ---------
        name : str
            The name of the expression
        expr : sympy.Basic, scalar
            The expression
        dependent : gotran.ODEObject
            If given the count of this expression will follow as a
            fractional count based on the count of the dependent object
        """

        # Create an Intermediate in the present component
        timer = Timer("Add intermediate")  # noqa: F841
        expr = Intermediate(name, expr, dependent)

        self._register_component_object(expr, dependent)

        return expr.sym

    def add_derivative(self, der_expr, dep_var, expr, dependent=None):
        """
        Add a derivative expression

        Arguments
        ---------
        der_expr : gotran.Expression, gotran.State, sympy.AppliedUndef
            The Expression or State which is differentiated
        dep_var : gotran.State, gotran.Time, gotran.Expression, sympy.AppliedUndef, sympy.Symbol
            The dependent variable
        expr : sympy.Basic
            The expression which the differetiation should be equal
        dependent : gotran.ODEObject
            If given the count of this expression will follow as a
            fractional count based on the count of the dependent object
        """
        timer = Timer("Add derivatives")  # noqa: F841

        if isinstance(der_expr, AppliedUndef):
            name = sympycode(der_expr)
            der_expr = self.root.present_ode_objects.get(name)

            if der_expr is None:
                error(f"{name} is not registered in this ODE")

        if isinstance(dep_var, (AppliedUndef, sp.Symbol)):
            name = sympycode(dep_var)
            dep_var = self.root.present_ode_objects.get(name)

            if dep_var is None:
                error(f"{name} is not registered in this ODE")

        # Check if der_expr is a State
        if isinstance(der_expr, State):
            self._expect_state(der_expr)
            obj = StateDerivative(der_expr, expr, dependent)

        else:

            # Create a DerivativeExpression in the present component
            obj = DerivativeExpression(der_expr, dep_var, expr, dependent)

        self._register_component_object(obj, dependent)

        return obj.sym

    def add_algebraic(self, state, expr, dependent=None):
        """
        Add an algebraic expression which relates a State with an
        expression which should equal to 0

        Arguments
        ---------
        state : gotran.State
            The State which the algebraic expression should determine
        expr : sympy.Basic
            The expression that should equal 0
        dependent : gotran.ODEObject
            If given the count of this expression will follow as a
            fractional count based on the count of the dependent object
        """

        state = self._expect_state(state)

        if f"d{state.name}_dt" in self.ode_objects:
            error(
                "Cannot registered an algebraic expression for a state "
                "that has a state derivative registered.",
            )

        if state.is_solved:
            error(
                "Cannot registered an algebraic expression for a state "
                "which is registered solved.",
            )

        # Create an AlgebraicExpression in the present component
        obj = AlgebraicExpression(state, expr, dependent)

        self._register_component_object(obj, dependent)

    @property
    def states(self):
        """
        Return a list of all states in the component and its children
        """
        return [state for state in iter_objects(self, False, False, False, State)]

    @property
    def full_states(self):
        """
        Return a list of all states in the component and its children that are
        not solved and determined by a state expression
        """
        return [
            expr.state for expr in self.state_expressions if not expr.state.is_solved
        ]

    @property
    def full_state_vector(self):
        """
        Return a sympy column vector with all full states
        """
        states = self.full_states

        return sp.Matrix(len(states), 1, lambda i, j: states[i].sym)

    @property
    def parameters(self):
        """
        Return a list of all parameters in the component
        """
        return ode_objects(self, Parameter)

    @property
    def intermediates(self):
        """
        Return a list of all intermediates
        """
        return [obj for obj in iter_objects(self, False, False, False, Intermediate)]

    @property
    def state_expressions(self):
        """
        Return a list of state expressions
        """
        return sorted(
            (obj for obj in iter_objects(self, False, False, False, StateExpression))
        )

    @property
    def rate_expressions(self):
        """
        Return a list of rate expressions
        """
        return [obj for obj in iter_objects(self, False, False, False, RateExpression)]

    @property
    def components(self):
        """
        Return a list of all child components in the component
        """
        return ode_components(self)

    @property
    def comments(self):
        return [com for comp in ode_components(self) for com in comp._local_comments]

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
    def num_parameters(self):
        """
        Return the number of all parameters
        """
        return len(self.parameters)

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
    def num_rate_expressions(self):
        """
        Return the number rate expressions
        """
        return len(self.rate_expressions)

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
        return self.is_locally_complete and all(
            child.is_complete for child in list(self.children.values())
        )

    @property
    def is_locally_complete(self):
        """
        True if the number of non-solved states are the same as the number
        of registered state expressions
        """
        num_local_states = sum(
            1
            for obj in self.ode_objects
            if isinstance(obj, State) and not obj.is_solved
        )

        return num_local_states == len(self._local_state_expressions)

    def __setattr__(self, name, value):
        """
        A magic function which will register expressions and simpler
        state expressions
        """

        from modelparameters.sympytools import symbols_from_expr

        # If we are registering a protected attribute or an attribute
        # during construction, just add it to the dict
        if name[0] == "_" or not self._allow_magic_attributes:
            self.__dict__[name] = value
            return

        # If no expression is registered
        if (not isinstance(value, scalars + (sp.Number,))) and not (
            isinstance(value, sp.Basic) and symbols_from_expr(value)
        ):
            debug(
                "Not registering: {0} as attribut. It does not contain "
                "any symbols or scalars.".format(name),
            )

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

            else:
                self.add_derivative(expr_obj, var_obj, value)

        elif TYPE == STATE_SOLUTION_EXPRESSION:
            self.add_state_solution(expr, value)

        elif TYPE == ALGEBRAIC_EXPRESSION:

            # Try getting corresponding ODEObjects
            (var_name,) = expr.groups()
            var_obj = self.root.present_ode_objects.get(var_name)

            if var_obj is None:
                self.add_intermediate(name, value)
            else:
                self.add_algebraic(var_obj, value)

    def __call__(self, name):
        """
        Return a child component, if the component does not excist, create
        and add one
        """
        check_arg(name, str)

        # If the requested component is the root component
        if self == self.root and name == self.root.name:
            comp = self.root
        else:
            comp = self.children.get(name)

        if comp is None:
            comp = self.add_component(name)
            debug(f"Adding '{name}' component to {self}")
        else:
            self.root._present_component = comp.name

        return comp

    def _expect_state(self, state, allow_state_solution=False, only_local_states=False):
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
            state = self.root.present_ode_objects.get(name)

            if state is None:
                error(f"{name} is not registered in this ODE")

            if only_local_states and not (
                state in self.states
                or (state in self.intermediates and allow_state_solution)
            ):
                error(f"{name} is not registered in component {self.name}")

        check_arg(state, allowed, 0)

        if isinstance(state, State) and state.is_solved:
            error(
                "Cannot registered a state expression for a state "
                "which is registered solved.",
            )

        return state

    def _register_component_object(self, obj, dependent=None):
        """
        Register an ODEObject to the component
        """

        if self._is_finalized:
            error(
                "Cannot add {0} {1} to component {2} it is "
                "already finalized.".format(obj.__class__.__name__, obj, self),
            )

        self._check_reserved_wordings(obj)

        # Register the object in the root ODE,
        # (here all duplication checks and expression expansions are done)
        self.root.register_ode_object(obj, self, dependent)

        # If registering a StateExpression
        if isinstance(obj, StateExpression):

            if self.rates and not self._is_finalizing:
                error(
                    "A component cannot have both state expressions "
                    "(derivative and algebraic expressions) and rate "
                    "expressions.",
                )

            if obj.state in self._local_state_expressions:
                error(
                    "A StateExpression for state {0} is already registered "
                    "in this component.".format(obj.state.name),
                )

            # Check that the state is registered in this component
            state_obj = self.ode_objects.get(obj.state.name)
            if not isinstance(state_obj, State):
                error(
                    "The state expression {0} defines state {1}, which is "
                    "not registered in the {2} component.".format(obj, obj.state, self),
                )

            self._local_state_expressions[obj.state] = obj

        # If obj is Intermediate register it as an attribute so it can be used
        # later on.
        if isinstance(obj, (State, Parameter, IndexedObject, Expression)):

            # If indexed expression or object register the basename as a dict
            if isinstance(obj, (IndexedExpression, IndexedObject)):
                if obj.basename in self.__dict__ and isinstance(
                    self.__dict__[obj.basename],
                    dict,
                ):
                    self.__dict__[obj.basename][obj.indices] = obj.sym
                else:
                    # FIXME: Initialize unused indices with zero
                    self.__dict__[obj.basename] = {obj.indices: obj.sym}

            # Register symbol, overwrite any already excisting symbol
            else:
                self.__dict__[obj.name] = obj.sym

        # Register the object
        self.ode_objects.append(obj)

    def _check_reserved_wordings(self, obj):
        if obj.name in _all_keywords:
            error(
                "Cannot register a {0} with a computer language "
                "keyword name: {1}".format(obj.__class__.__name__, obj.name),
            )

        # Check for reserved Expression wordings
        # if isinstance(obj, Expression):
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

    def _add_rates(self, states, rate_matrix):
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

        check_arg(states, (tuple, list), 0, ODEComponent._add_rates)
        check_arg(rate_matrix, sp.MatrixBase, 1, ODEComponent._add_rates)

        # If list
        if isinstance(states, list):
            states = (states, states)

        # else tuple
        elif len(states) != 2 and not all(
            isinstance(list_of_states, list) for list_of_states in states
        ):
            error("expected a tuple of 2 lists with states as the " "states argument")

        # Check index arguments
        # for list_of_states in states:
        #    print list_of_states, local_states
        #    if not all(state in local_states for state in list_of_states):
        #        error("Expected the states arguments to be States in "\
        #              "the Markov model")

        # Check that the length of the state lists corresponds with the shape of
        # the rate matrix
        if rate_matrix.shape[0] != len(states[0]) or rate_matrix.shape[1] != len(
            states[1],
        ):
            error("Shape of rates does not match given states")

        for i, state_i in enumerate(states[0]):
            for j, state_j in enumerate(states[1]):
                value = rate_matrix[i, j]

                # If 0 as rate
                if (isinstance(value, scalars) and value == 0) or (
                    isinstance(value, sp.Basic) and value.is_zero
                ):
                    continue

                if state_i == state_j:
                    error("Cannot have a nonzero rate value between the " "same states")

                # Assign the rate
                self._add_single_rate(state_i, state_j, value)

    def _add_single_rate(self, to_state, from_state, expr):
        """
        Add a single rate expression
        """

        if self.state_expressions:
            error(
                "A component cannot have both state expressions (derivative "
                "and algebraic expressions) and rate expressions.",
            )

        check_arg(expr, scalars + (sp.Basic,), 2, ODEComponent._add_single_rate)

        expr = sp.sympify(expr)

        to_state = self._expect_state(
            to_state,
            allow_state_solution=True,
            only_local_states=True,
        )
        from_state = self._expect_state(
            from_state,
            allow_state_solution=True,
            only_local_states=True,
        )

        if to_state == from_state:
            error("The two states cannot be the same.")

        if (to_state.sym, from_state.sym) in self.rates:
            error(
                f"Rate between state {from_state} and {to_state} is already registered.",
            )

        # FIXME: It should also not be possible to include other
        # states in the markov model, right?
        syms_expr = symbols_from_expr(expr)
        if to_state.sym in syms_expr or from_state.sym in syms_expr:
            error(
                "The rate expression cannot be dependent on the " "states it connects.",
            )

        # Create a RateExpression
        obj = RateExpression(to_state, from_state, expr)

        self._register_component_object(obj)

        # Store the rate sym in the rate dict
        self.rates._register_single_rate(to_state.sym, from_state.sym, obj.sym)

    def finalize_component(self):
        """
        Called whenever the component should be finalized
        """
        if self._is_finalized:
            return

        # A flag to allow adding of for example Markov model rates
        self._is_finalizing = True

        # If component is a Markov model
        if self.rates:
            self._finalize_markov_model()

        if not self.is_locally_complete:
            incomplete_states = []
            for obj in self.ode_objects:
                if isinstance(obj, State):
                    if obj not in self._local_state_expressions:
                        incomplete_states.append(obj)

            incomplete_state_names = [s.name for s in incomplete_states]

            msg = f"Cannot finalize component '{self}'. "
            msg += f"Missing time derivatives for the following states: {', '.join(incomplete_state_names)}"

            error(msg)

        self._is_finalizing = False
        self._is_finalized = True

    def _finalize_markov_model(self):
        """
        Finalize a markov model
        """

        # Derivatives
        states = self.states
        derivatives = defaultdict(lambda: sp.sympify(0.0))
        rate_check = defaultdict(lambda: 0)

        # Build rate information and check that each rate is added in a
        # symetric way
        used_states = [0] * self.num_states
        for rate in self.rate_expressions:

            # Get the states
            to_state, from_state = rate.states

            # Add to derivatives of the two states
            derivatives[from_state] -= rate.sym * from_state.sym
            derivatives[to_state] += rate.sym * from_state.sym

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
            error(f"No rate registered for state {states[used_states.find(0)]}")

        # Check rate symetry
        for (ind_from, ind_to), times in list(rate_check.items()):
            if times != 2:
                error(
                    "Only one rate between the states {0} and {1} was "
                    "registered, expected two.".format(
                        states[ind_from],
                        states[ind_to],
                    ),
                )

        # Add derivatives
        for state in states:

            # Skip solved states
            if not isinstance(state, State) or state.is_solved:
                continue

            self.add_derivative(state, state.time.sym, derivatives[state])
