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

__all__ = ["ODEObjectList", "ODEBaseComponent", "PresentObjTuple"]

# System imports
from collections import OrderedDict
import weakref

from sympy.core.function import AppliedUndef

# ModelParameters imports
from modelparameters.sympytools import sp, symbols_from_expr
from modelparameters.utils import tuplewrap, Timer
from modelparameters.codegeneration import sympycode, _all_keywords

# Local imports
from gotran.common import error, debug, check_arg, check_kwarg, scalars
from gotran.model.odeobjects2 import *
from gotran.model.expressions2 import *

# Create a weak referenced tuple class
class PresentObjTuple(object):
    def __init__(self, obj, comp):
        self.obj = obj
        self._comp = weakref.ref(comp)

    @property
    def comp(self):
        return self._comp()

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

class ODEObjectList(list):
    """
    Specialized container for ODEObjects. It is a list but adds dict
    access through the name attribute of an ODEObjects
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
        self._parent = weakref.ref(parent)

        # Store ODEBaseComponent children
        self.children = OrderedDict()

        # Store ODEObjects of this component
        self.ode_objects = ODEObjectList()

        # Store all state expressions
        self._local_state_expressions = dict()

        # Collect all comments
        self._local_comments = []

        # Flag to check if component is finalized
        self._is_finalized = False

        self._constructed = True

    @property
    def parent(self):
        """
        Return the the parent if it is still alive. Otherwise it will return None
        """
        p = self._parent()
        if p is None:
            error("Cannot access parent ODEComponent. It is already "\
                  "destroyed.")
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
        comment = Comment(comment)
        self.ode_objects.append(comment)
        self._local_comments.append(comment)

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
        from odecomponents2 import DerivativeComponent

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
        from odecomponents2 import MarkovModelComponent
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
            der_expr = der_expr.obj

        if isinstance(dep_var, (AppliedUndef, sp.Symbol)):
            name = sympycode(dep_var)
            dep_var = self.root.present_ode_objects.get(name)

            if dep_var is None:
                error("{0} is not registered in this ODE".format(name))
            dep_var = dep_var.obj

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
        return [obj for obj in iter_objects(self, False, False, False, \
                                            Intermediate)]

    @property
    def state_expressions(self):
        """
        Return a list of state expressions
        """
        return sorted((obj for obj in iter_objects(self, False, False, \
                                                   False, StateExpression)),\
                      lambda o0, o1 : cmp(o0.state, o1.state))

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

            state, comp = state_comp.obj, state_comp.comp

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
        if isinstance(obj, (State, Parameter, IndexedObject, Expression)):

            # If indexed expression or object register the basename as a dict 
            if isinstance(obj, (IndexedExpression, IndexedObject)):
                if obj.basename in self.__dict__ and isinstance(\
                    self.__dict__[obj.basename], dict):
                    self.__dict__[obj.basename][obj.indices] = obj.sym
                else:
                    # FIXME: Initialize unused indices with zero
                    self.__dict__[obj.basename] = {obj.indices:obj.sym}

            # Register symbol, overwrite any already excisting symbol
            else:
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

