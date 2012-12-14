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

# Local imports
from gotran.common import error, check_arg, scalars
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

        # Store external dependencies
        self.external_object_dep = set()
        self.external_component_dep = set()

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

    

class MarkovModel(object):
    """
    A Markov model class
    """
    def __init__(self, name, ode, algrebraic_sum=None, **states):
        """
        Initalize a Markov model

        Arguments
        ---------
        name : str
            Name of Markov model
        algebraic_sum : scalar (optional)
            If the algebraic sum of all states should be constant,
            give the value here.
        states : dict
            A dict with all states defined in this Markov model
        """
        check_arg(name, str, 0)
        check_arg(ode, ODE, 1)

        if len(states) < 2:
            error("Expected at least two states in a Markov model")

        if algebraic_sum is not None:
            check_arg(algebraic_sum, scalars)
        self._algebraic_sum = algebraic_sum

        self._name = name
        self._ode = ode
        self._is_finalized = False

        # Check states kwargs
        state_sum = 0.0
        for name, init in states.items():
            check_kwargs(init, name, scalars)
            state_sum += init

        # Check algebraic sum agains initial values
        if self._algebraic_sum is not None:
            if state_sum != self._algebraic_sum:
                error("The given algebraic sum does not match the sum of "\
                      "the initial state values ")
            
            # Find the state which will be excluded from the states
            for name in states:
                if "O" not in name:
                    break

            algebraic_state = name
        else:
            algebraic_state = ""

        # Add states to ode
        collected_states = ODEObjectList()
        for name, init in states.items():

            # If we are not going to add the state
            if name == algebraic_state:
                continue
            
            sym = ode.add_state(name, init, component=name)
            collected_states.append(ode.get_object(sym))

        # Add an intermediate for the algebraic state
        if self._algebraic_sum is not None:
            sym = ode.add_intermediate(algebraic_state, 1 - reduce(\
                lambda x,y:x+y, (state.sym for state in collected_states), 0))
            inter_obj = ode.get_object(sym)
            collected_states.append(inter_obj)
        
        # Store state attributes
        self._states = collected_states

        # Rate attributes
        self._rates = {}

    def __setitem__(self, states, rate):
        """
        Set a rate between states given in states

        Arguments
        ---------
        states : tuple of size two
            A tuple of two states for which the rate is going to be between
        rate : scalar or sympy expression
            An expression of the rate between the two states
        """
        check_arg(states, tuple, 0, MarkovModel.__setitem__, int)

        if self._is_finalized:
            error("The Markov model is finalized. No more rates can be added.")
            
        if len(states) != 2:
            error("Expected the states argument to be a tuple of two states")

        if not all(state in self._states for state in states):
            error("Expected the states arguments to be States in "\
                  "the Markov model")
            
        if states[0] == states[1]:
            error("The two states cannot be the same.")

        if states in self._rates:
            error("Rate between state {0} and {1} is already "\
                  "registered.".format(*states))

        # Create an Expression of the rate and store it
        self._rate[states] = Expression("{0}-{1}".format(*states), rate)

        # Check that the states are not used in the rates
        for dep_obj in self._rate[states].object_dependencies:
            if dep_obj in self._states:
                error("Markov model rate cannot include state variables in the "\
                      "same Markov model: {0}".format(dep_obj))
    
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
        derivatives = OrderedDict((state, 0.0) for state in self._states)
        rate_check = {}

        # Build rate information and check that each rate is added in a
        # symetric way
        used_states = [0]*len(self._states)
        for states, rate in self._rates:
            from_state, to_state = states

            # Add to derivatives of the two states
            derivatives[from_state] -= rate*from_state
            derivatives[to_state] += rate*from_state

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
            if time != 2:
                error("Only one rate between the states {0} and {1} was "\
                      "registered, expected two.".format(\
                          self._states[ind_from], self._states[ind_to]))

        # Add derivatives
        for state in self._states:
            if isinstance(state, State):
                self._ode.diff(state.derivative.sym, derivatives[state])
        

    @property
    def is_finalized(self):
        return self._is_finalized

    @property
    def name(self):
        return self._name

    @property
    def num_states(self):
        return len(self._states)
