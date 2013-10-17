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
from modelparameters.sympytools import sp, iter_symbol_params_from_expr
from modelparameters.parameters import ScalarParam
from modelparameters.codegeneration import sympycode, _all_keywords

# Local imports
from gotran.common import error, check_arg, check_kwarg, scalars
from gotran.model.odeobjects import *

class Comment(ODEObject):
    """
    A Comment class. To keep track of user comments in an ODE
    """
    def __init__(self, comment):
        """
        Create a comment 

        Arguments
        ---------
        comment : str
            The comment
        """
        
        # Call super class
        super(Comment, self).__init__(comment)

class ODEComponent(ODEObject):
    """
    A Component class. To keep track of Components in an ODE
    """
    def __init__(self, name, ode):
        """
        Create an ODEComponent

        Arguments
        ---------
        name : str 
            The name of the component. This str serves as the unique
            identifier of the Component.
        ode : ODE
            The ode object of the component
        """

        from gotran.model.ode2 import ODE
        check_arg(name, str, 0, ODEComponent)
        check_arg(ode, ODE, 1, ODEComponent)
        
        # Call super class
        super(ODEComponent, self).__init__(name)

        # Store ode
        self._ode = ode

        # Store ODEObjects of this component
        self.ode_objects = ODEObjectList()

        # Store external dependencies
        self.external_object_dep = OrderedDict()
        self.external_component_dep = OrderedDict()

    def state(self, name, init):
        """
        Add a state to the component

        Arguments
        ---------
        name : str
            The name of the state variable
        init : scalar, ScalarParam
            The initial value of the state
        """

        # Create state
        state = State(name, init)

        # FIXME: Create this inside State constructor?
        state_der = StateDerivative(state)
        state.derivative = state_der

        self._register_object(state)
        self._register_object(state_der)

        # Return the sympy version of the state
        return state.sym

    def parameter(self, name, init):
        """
        Add a parameter to the component

        Arguments
        ---------
        name : str
            The name of the parameter
        init : scalar, ScalarParam
            The initial value of the parameter
        """
        
        param = Parameter(name, init)

        self._register_object(param)

        # Return the sympy version of the state
        return param.sym

    def variable(self, name, init):
        """
        Add a variable to the component

        Arguments
        ---------
        name : str
            The name of the variable
        init : scalar, ScalarParam
            The initial value of the variable
        """
        
        variable = Variable(name, init)

        self._register_object(variable)

        # Return the sympy version of the state
        return variable.sym

    def derivative(self, sym, expr):
        pass

    def algebraic(self, sym, expr):
        pass

    def rates(self, states, rates):
        pass

    def expression(self, name, expr):
        """
        Register a math expression

        Arguments
        ---------
        name : str
            The name of the expression
        expr : sympy.Basic, scalar
            The expression
        """

        # Create an Expression in the present component
        expr = Expression(name, expr)

        # Iterate over dependencies in the expression
        intermediate_objects = []
        object_dependencies = ODEObjectList()
        for sym in iter_symbol_params_from_expr(expr):
            dep_obj = ode.get_object(sym) or ode._intermediates.get(sym)
            if dep_obj is None:
                error("The symbol '{0}' is not declared within the '{1}' "\
                      "ODE.".format(sym, ode.name))

            # Store object dependencies
            object_dependencies.append(dep_obj)
            
            # Check that we are not using a DerivativeExpressions in expression
            if isinstance(dep_obj, (StateDerivative, DerivativeExpression)):
                error("An expression cannot include a StateDerivative or "\
                      "DerivativeExpression")

            # Collect intermediates to be used in substitutions below
            if isinstance(dep_obj, Intermediate):
                intermediate_objects.append(dep_obj)


        expr.expanded_expr = expr.sym.subs((dep_obj.sym, dep_obj.expanded_expr) \
                                           for dep_obj in intermediate_objects)

        self._object_dependencies = object_dependencies

    def _register_object(self, obj):
        """
        Register an ODEObject to the component
        """
        
        if obj.name in _all_keywords:
            error("Cannot register a {0} with a computer language "\
                  "keyword name: {1}".format(obj.__class__.__name__,
                                             obj.name))

        # Check for reserved wording of StateDerivatives
        if re.search(_derivative_name_template, obj.name):
            error("The pattern d{{name}}_dt is reserved for derivatives. "
                  "However {0} is not a state derivative.".format(\
                      intermediate.name))
        
        def duplication_error(obj, dup_obj):
            error("Cannot register a {0}. A {1} with name '{2}' is "\
                  "already registered in this ODE.".format(\
                      type(obj).__name__, type(dup_obj).__name__, dup_obj.name))

        # Check for existing object in the ODE
        duplication = self.ode.get_object(obj.name, return_component=True)

        # If object with same name is already registered in the ode we
        # need to figure out what to do
        if duplication:

            dup_obj, comp = duplication

            # If a parameter or Variable is registered using the same name
            # as the ODE it can be overwritten with the new object if the new object 
            if isinstance(dub_obj, (Parameter, Variable)) and \
                   comp.name == self.ode.name:
                
                
            # If State we always raise an error
            if isinstance(dup_obj, State):
                duplication_error(obj, dup_obj)
                
            if isinstance(dub_obj, StateDerviative):
                
                
        # Check for existing object
        dup_obj = self.ode_objects.get(name)
        
        # If the object already exists and is a StateDerivative
        if dup_obj is not None and isinstance(dup_obj, StateDerivative):
            
            self.derivative(dup_obj.sym, expr)
            return
        
        # Check for duplicates
        if intermediate.name in self._intermediates:
            self._intermediate_duplicates.add(intermediate.name)

        # Store the intermediate
        self._intermediates.append(intermediate)

        # Add to component
        self._present_component.append(intermediate)

        # Register symbol, overwrite any already excisting symbol
        self.__dict__[name] = intermediate.sym

        # Return symbol
        return intermediate.sym

                # Logic to handle state derivatives

        # Register the object 
        self.ode_objects[obj.name] = obj

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
        super(MarkovModel, self).__init__(name, component)

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

        if all(isinstance(state, sp.Symbol) for state in states):
            
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
