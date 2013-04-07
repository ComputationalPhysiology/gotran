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

__all__ = ["ODEObjectList", "ODEComponent", "Comment", "MarkovModel"]

# System imports
from collections import OrderedDict

# ModelParameters imports
from modelparameters.sympytools import sp, ModelSymbol, \
     iter_symbol_params_from_expr
from modelparameters.parameters import ScalarParam

# Local imports
from gotran.common import error, check_arg, check_kwarg, scalars
from gotran.model.odeobjects import *

class Comment(ODEObject):
    """
    A Comment class. To keep track of user comments in an ODE
    """
    def __init__(self, comment, component=""):
        """
        Create a comment 

        Arguments
        ---------
        comment : str
            The comment
        component : str (optional)
            A component for which the Comment should be associated with.
        """
        
        # Call super class
        super(Comment, self).__init__(comment, component, "")

class ODEComponent(ODEObject):
    """
    A Component class. To keep track of Components in an ODE
    """
    def __init__(self, component, ode):
        """
        Create an ODEComponent

        Arguments
        ---------
        component : str 
            The name of the component. This str serves as the unique
            identifier of the Component.
        ode : ODE
            The ode object of the component
        """

        from gotran.model.ode import ODE
        check_arg(component, str, 0, ODEComponent)
        check_arg(ode, ODE, 1, ODEComponent)
        
        # Call super class
        super(ODEComponent, self).__init__(component, "", "")

        # Store ode
        self._ode = ode

        # Store ODEObjects of this component
        self.states = OrderedDict()
        self.parameters = OrderedDict()
        self.variables = OrderedDict()
        self.intermediates = ODEObjectList()
        self.derivatives = ODEObjectList()
        self.markov_models = ODEObjectList()

        # Store external dependencies
        self.external_object_dep = set()
        self.external_component_dep = set()

    def remove(self, obj):
        """
        Remove an ODEObject from the Component
        """
        assert(isinstance(obj, (State, Parameter, Variable)))
        if isinstance(obj, State):
            self.states.pop(obj.name)
        elif isinstance(obj, Parameter):
            self.parameters.pop(obj.name)
        elif isinstance(obj, Variable):
            self.variables.pop(obj.name)

        # Remove object and component dependencies
        for comp in self._ode.components.values():
            if obj not in comp.external_object_dep:
                continue
            comp.external_object_dep.remove(obj)

            # If not same we need to check if external component
            # dependency still applies
            for other_obj in comp.external_object_dep:
                if obj.component == other_obj.component:
                    break
            else:
                # No rest of old component left in external_object_dep
                assert(obj.component in comp.external_component_dep)
                comp.external_component_dep.remove(obj.component)

    def append(self, obj):
        """
        Append an ODEObject to the Component
        """
        from gotran.model.expressions import Expression, Intermediate, \
             DerivativeExpression
        
        check_arg(obj, ODEObject, 0, ODEComponent.append)

        assert(obj.component == self.name)

        # If SingleODEObject we need to check that no Expressions has been added
        if isinstance(obj, SingleODEObjects):

            #if (len(self.intermediates) + len(self.derivatives))>0:
            #    error("Cannot register a {0} to '{1}' after an expression"\
            #          " has been register.".format(\
            #              type(obj).__name__, self.name, ))

            # Store object
            if isinstance(obj, State):
                self.states[obj.name] = obj
            elif isinstance(obj, Parameter):
                self.parameters[obj.name] = obj
            elif isinstance(obj, Variable):
                self.variables[obj.name] = obj
            else:
                error("Not recognised SingleODEObject: {0}".format(\
                    type(obj).__name__))

        elif isinstance(obj, Expression):

            # If Intermediate we need to check that no DerivativeExpression
            # has been added
            if isinstance(obj, Intermediate):
                if self.derivatives:
                    error("Cannot register an Intermediate after"\
                          "a DerivativeExpression has been registered.")
            
            if isinstance(obj, Intermediate):
                self.intermediates.append(obj)
            elif isinstance(obj, DerivativeExpression):
                self.derivatives.append(obj)
            else:
                error("Not recognised Expression: {0}".format(\
                    type(obj).__name__))

            own_obj = self.states.keys() + self.parameters.keys() + \
                      self.variables.keys()

            # We have an expression and we need to figure out dependencies
            for sym in iter_symbol_params_from_expr(obj.expr):
                dep_obj = self._ode.get_object(sym) or \
                          self._ode.get_intermediate(sym)
                assert(dep_obj)
                assert(not isinstance(dep_obj, DerivativeExpression))

                if dep_obj in self.intermediates:
                    continue

                if dep_obj.name in own_obj:
                    continue
                
                self.external_object_dep.add(dep_obj)
                self.external_component_dep.add(dep_obj.component)

        elif isinstance(object, Comment):
            self.intermediates.append(obj)

        elif isinstance(obj, MarkovModel):
            self.markov_models.append(obj)
        else:
            error("Not recognised ODEObject: {0}".format(\
                type(obj).__name__))

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
        elif isinstance(name, ModelSymbol):
            return self._objects.get(name.name)
        return None
        
    def __contains__(self, item):
        if isinstance(item, str):
            return any(item == obj.name for obj in self)
        elif isinstance(item, ModelSymbol):
            return any(item.name == obj.name for obj in self)
        elif (item, ODEObject):
            return super(ODEObjectList, self).__contains__(item)
        return False

    def count(self, item):
        if isinstance(item, str):
            return sum(item == obj.name for obj in self)
        elif isinstance(item, ModelSymbol):
            return sum(item.name == obj.name for obj in self)
        elif (item, ODEObject):
            return super(ODEObjectList, self).count(item)
        return 0

    def index(self, item):
        if isinstance(item, str):
            for ind, obj in enumerate(self):
                if item == obj.name:
                    return ind
        elif isinstance(item, ModelSymbol):
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

    

class MarkovModel(ODEObject):
    """
    A Markov model class
    """
    def __init__(self, name, ode, component="", *args, **kwargs):
        """
        Initialize a Markov model

        Arguments
        ---------
        name : str
            Name of Markov model
        ode : ODE
            The ode the Markov Model should be added to
        component : str (optional)
            Add state to a particular component
        algebraic_sum : scalar (optional)
            If the algebraic sum of all states should be constant,
            give the value here.
        args : list of tuples
            A list of tuples with states and init values. Use this to set states
            if you need them ordered.
        kwargs : dict
            A dict with all states defined in this Markov model
        """

        from ode import ODE

        check_arg(ode, ODE, 1)

        # Call super class
        super(MarkovModel, self).__init__(name, component, ode.name)

        algebraic_sum = kwargs.pop("algebraic_sum", None)
        states = list(args) + sorted(kwargs.items())

        if len(states) < 2:
            error("Expected at least two states in a Markov model")

        if algebraic_sum is not None:
            check_arg(algebraic_sum, scalars, gt=0.0)
        
        self._algebraic_sum = algebraic_sum
        self._algebraic_expr = None
        self._algebraic_name = None

        self._ode = ode
        self._is_finalized = False

        # Check states kwargs
        state_sum = 0.0
        for state_name, init in states:
            # FIXME: Allow Parameter as init
            check_kwarg(init, state_name, scalars + (ScalarParam,))
            state_sum += init if isinstance(init, scalars) else init.value

        # Check algebraic sum agains initial values
        if self._algebraic_sum is not None:
            if abs(state_sum - self._algebraic_sum) > 1e-8 :
                error("The given algebraic sum does not match the sum of "\
                      "the initial state values: {0}!={1}.".format(\
                          self._algebraic_sum, state_sum))
            
            # Find the state which will be excluded from the states
            for state_name, init in states:
                if "O" not in state_name:
                    break

            algebraic_name = state_name
        else:
            algebraic_name = ""

        # Add states to ode
        collected_states = ODEObjectList()
        for state_name, init in states:

            # If we are not going to add the state
            if state_name == algebraic_name:
                continue

            # Add a slaved state
            sym = ode.add_state(state_name, init, component=component, slaved=True)
            collected_states.append(ode.get_object(sym))

        # Add an intermediate for the algebraic state
        if self._algebraic_sum is not None:

            algebraic_expr = 1 - reduce(lambda x,y:x+y, \
                                (state.sym for state in collected_states), 0)
            
            # Add a slaved intermediate
            sym = ode.add_intermediate(algebraic_name, algebraic_expr, \
                                       component=component, slaved=True)

            collected_states.append(ode.intermediates.get(sym))

        # Store state attributes
        self._states = collected_states

        # Rate attributes
        self._rates = OrderedDict()

    def __setitem__(self, states, rate):
        """
        Set a rate between states given in states

        Arguments
        ---------
        states : tuple of size two
            A tuple of two states for which the rate is going to be between,
            a list of states is also accepted. If two lists are passed something
            with a shape is expected as rate argument.
        rate : scalar or sympy expression
            An expression of the rate between the two states
        """
        from gotran.model.expressions import Expression

        check_arg(states, (tuple, list), 0, MarkovModel.__setitem__)

        if self._is_finalized:
            error("The Markov model is finalized. No more rates can be added.")

        # If given one list of states as arguments we assume the a quadratic
        # assignment of rates.
        if isinstance(states, list):
            states = (states, states)
            
        if len(states) != 2:
            error("Expected the states argument to be a tuple of two states")

        if all(isinstance(state, ModelSymbol) for state in states):
            
            if not all(state in self._states for state in states):
                error("Expected the states arguments to be States in "\
                      "the Markov model")

        elif all(isinstance(state, list) for state in states):

            # Check index arguments
            for list_of_states in states:
                if not all(state in self._states for state in list_of_states):
                    error("Expected the states arguments to be States in "\
                          "the Markov model")

            # Check rate matrix
            if not hasattr(rate, "shape"):
                error("When passing list of states as indices a rate with "\
                      "shape attribute is expected.")

            # Assign the individual rates
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
                    self[state_i, state_j] = value

            # Do not continue after matrix is added
            return
                
        else:
            error("Expected either a list of states or only states as indices.")
            
        if states[0] == states[1]:
            error("The two states cannot be the same.")

        if states in self._rates:
            error("Rate between state {0} and {1} is already "\
                  "registered.".format(*states))

        # Create an Expression of the rate and store it
        self._rates[states] = Expression("{0}-{1}".format(*states), rate, \
                                         self._ode)

        # Check that the states are not used in the rates
        for dep_obj in self._rates[states].object_dependencies:
            if dep_obj in self._states:
                error("Markov model rate cannot include state variables "\
                      "in the same Markov model: {0}".format(dep_obj))
    
    def finalize(self):
        """
        Finalize the Markov model.

        This will add the derivatives to the ode model. After this is
        done no more rates can be added to the Markov model.
        """
        if self._is_finalized:
            return
        
        self._is_finalized = True

        # Derivatives
        derivatives = OrderedDict((state.sym, 0.0) for state in self._states)
        rate_check = {}

        # Build rate information and check that each rate is added in a
        # symetric way
        used_states = [0]*len(self._states)
        for (from_state, to_state), rate in self._rates.items():

            # Add to derivatives of the two states
            derivatives[from_state] -= rate.expr*from_state
            derivatives[to_state] += rate.expr*from_state

            # Register rate
            ind_from = self._states.index(from_state)
            ind_to = self._states.index(to_state)
            ind_tuple = (min(ind_from, ind_to), max(ind_from, ind_to))
            if ind_tuple not in rate_check:
                rate_check[ind_tuple] = 0
            rate_check[ind_tuple] += 1

            used_states[ind_from] = 1
            used_states[ind_to] = 1

        # Check used states
        if 0 in used_states:
            error("No rate registered for state {0}".format(\
                self._states[used_states.find(0)]))

        # Check rate symetry
        for (ind_from, ind_to), times in rate_check.items():
            if times != 2:
                error("Only one rate between the states {0} and {1} was "\
                      "registered, expected two.".format(\
                          self._states[ind_from], self._states[ind_to]))

        # Add derivatives
        for state in self._states:
            if isinstance(state, State):
                self._ode.diff(state.derivative.sym, derivatives[state.sym], \
                               self.component)
        

    @property
    def is_finalized(self):
        return self._is_finalized

    @property
    def num_states(self):
        return len(self._states)

    @property
    def states(self):
        return self._states

    def _set_algebraic_sum(self, value):
        check_arg(value, scalars, gt=0)
        _algebraic_sum = value
