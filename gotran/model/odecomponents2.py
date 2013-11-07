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

__all__ = ["ODE", "ODEObjectList", "ODEComponent", "Comment", "MarkovModel"]

# System imports
from collections import OrderedDict, defaultdict
import re

from sympy.core.function import AppliedUndef

# ModelParameters imports
from modelparameters.sympytools import sp, symbols_from_expr, iter_symbol_params_from_expr
from modelparameters.parameters import ScalarParam
from modelparameters.parameterdict import ParameterDict
from modelparameters.codegeneration import sympycode, _all_keywords

# Local imports
from gotran.common import error, check_arg, check_kwarg, scalars
from gotran.model.odeobjects2 import *

_derivative_name_template = re.compile("\Ad([a-zA-Z]\w*)_d([a-zA-Z]\w*)\Z")

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

class iter_objects(object):
    """
    A recursive iterator over all objects of a component including its childrens

    Arguments
    ---------
    comp : ODEComponent
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
    def __init__(self, comp, reverse=False, *types):
        assert isinstance(comp, ODEComponent)
        if reverse:
            self._object_iterator = self._reverse_iter_objects(comp)
        else:
            self._object_iterator = self._iter_objects(comp)
        self._types = types or ODEObject
        assert all(issubclass(T, ODEObject) for T in self._types)

    def _reverse_iter_objects(self, comp):

        # First all children components in reversed order
        for sub_comp in reversed(comp.children.values()):
            for sub_tree in self._reverse_iter_objects(sub_comp):
                yield sub_tree
        
        # Secondly return component
        yield comp

        # Last all objects
        for obj in reversed(comp.ode_objects):
            if isinstance(obj, self._types):
                yield obj


    def _iter_objects(self, comp):

        # First return component
        yield comp

        # Secondly all objects
        for obj in comp.ode_objects:
            if isinstance(obj, self._types):
                yield obj

        # Thrirdly all children components
        for sub_comp in comp.children.values():
            for sub_tree in self._iter_objects(sub_comp):
                yield sub_tree

    def __next__(self):
        return next(self._object_iterator)

    def __iter__(self):
        return self


def _bubble_append(components, obj):
    """
    Help function to append an object to a list. If the object already
    excist in the list it will be moved to the end.
    """

    assert(isinstance(components, list))
    
    # If first component
    if len(components) == 0:
        components.append(obj)
                
    # If the component is alread the last one
    elif components[-1] == obj:
        pass

    # If the component is already registered 
    else:
        # Just remove the component in list if it is there
        try:
            components.remove(obj)
        except:
            # If not we do not have to do anything
            pass

        components.append(obj)


class ODEComponent(ODEObject):
    """
    A Component class. To keep track of Components in an ODE
    """
    def __init__(self, name, parent):
        """
        Create an ODEComponent

        Arguments
        ---------
        name : str 
            The name of the component. This str serves as the unique
            identifier of the Component.
        parent : ODEComponent
            The parent component of this ODEComponent
        """

        self._constructed = False
        check_arg(name, str, 0, ODEComponent)
        check_arg(parent, ODEComponent, 1, ODEComponent)
        
        # Call super class
        super(ODEComponent, self).__init__(name)

        # Store parent component
        self.parent = parent

        # Store ODEComponent children
        self.children = OrderedDict()

        # Store ODEObjects of this component
        self.ode_objects = ODEObjectList()

        self._constructed = True

    def get_object(self, name, reversed=True, return_component=False):
        comp, obj = None, None

        # If a name is registered
        if name in self.root.ns:

            for obj in iter_objects(self, reversed=reversed):
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

        # Create state
        state = State(name, init, self.time)

        self._register_component_object(state)

        # Return the sympy version of the state
        return state.sym

    def add_states(self, **kwargs):
        """
        Add a number of states to the current ODEComponent
        """
    
        if not kwargs:
            error("expected at least one state")
        
        # Symbol and value dicts
        for name, value in sorted(kwargs.items()):
    
            # Add the states
            self.add_state(name, value)

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
        
        param = Parameter(name, init)

        self._register_component_object(param)

        # Return the sympy version of the state
        return param.sym

    def add_parameters(self, **kwargs):
        """
        Add a number of parameters to the current ODEComponent
        """
    
        if not kwargs:
            error("expected at least one parameter")
        
        # Symbol and value dicts
        ns = {}
        for name, value in sorted(kwargs.items()):
    
            # Add the Parameter
            self.add_parameter(name, value)

    def add_component(self, name):
        """
        Add a sub ODEComponent
        """
        comp = ODEComponent(name, self)

        self.children[name] = comp

        return comp

    def add_derivative(self, der_expr, dep_var, expr):
        """
        Add a derivative expression 

        Arguments
        ---------
        der_expr : Expression, State
            The Expression or State which is differentiated
        dep_var : State, Time, Expression
            The dependent variable
        expr : sympy.Basic
            The expression which the differetiation should be equal
        """
        
        # Check that States being differentiated is contained in the
        # same component
        if isinstance(der_expr, State) and der_expr not in self.ode_objects:
            error("The state being differentiated need to be in "\
                  "the same component as the derivative.")
            
        # Create a DerivativeExpression in the present component
        expr = DerivativeExpression(der_expr, dep_var, expr)
        
        self._register_component_object(expr)

        return expr.sym

    def add_algebraic(self, sym, expr):
        pass

    def add_rates(self, states, rates):
        pass

    def add_expression(self, name, expr):
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

        self._register_component_object(expr)

        return expr.sym

    @property
    def root(self):
        """
        Return the root ODE component (the ode)
        """
        present = self
        while present != present.parent:
            present = present.parent

        return present

    def __setattr__(self, name, value):
        """
        A magic function which will register expressions and simpler
        derivative expressions
        """

        # If we are registering a protected attribute or an attribute
        # during construction, just add it to the dict
        if name[0] == "_" or not self._constructed:
            self.__dict__[name] = value
            return

        # If no expression is regostered
        if (not isinstance(value, scalars)) and not (isinstance(value, sp.Basic) \
                                                     and symbols_from_expr(value)):
            debug("Not registering: {0} as attribut. It does not contain "\
                  "any symbols or scalars.".format(name))

            # FIXME: Should we raise an error?
            return 

        # If not registering a derivative expression
        der_expr = re.search(_derivative_name_template, name)
        if not der_expr:
            self.add_expression(name, value)
            return 
            
        # Try getting corresponding ODEObjects
        expr_name, var_name = der_expr.groups()
        expr_obj = self.root.present_ode_objects.get(expr_name)
        var_obj = self.root.present_ode_objects.get(var_name)

        # If the expr or variable is not declared in this ODE
        if expr_obj is None:
            error("Trying to register a DerivativeExpression, but "\
                  "the expression: '{0}' is not registered in this "\
                  "ODE.".format(expr_name))

        if var_obj is None:
            error("Trying to register a DerivativeExpression, but "\
                  "the variable: '{0}' is not registered in this "\
                  "ODE.".format(var_name))
            
        self.add_derivative(expr_obj[0], var_obj[0], value)

        
    def __getitem__(self, name):
        """
        Return a child component
        """
        check_arg(name, str)
        comp = self.children.get(name)
        if comp is None:
            error("'{0}' is not a sub component of {1}".format(name, self))

        return comp

    def _register_component_object(self, obj):
        """
        Register an ODEObject to the component
        """
        
        if obj.name in _all_keywords:
            error("Cannot register a {0} with a computer language "\
                  "keyword name: {1}".format(obj.__class__.__name__,
                                             obj.name))

        # Check for reserved wording of DerivativeExpressions
        if not isinstance(obj, DerivativeExpression) and \
               re.search(_derivative_name_template, obj.name):
            error("The pattern d{{name}}_dt is reserved for derivatives. "
                  "However {0} is not a state derivative.".format(\
                      obj.name))
        
        # Register symbol, overwrite any already excisting symbol
        self.__dict__[obj.name] = obj.sym

        # Register the object in the root ODE,
        # (here all duplication checks are done)
        self.root.register_ode_object(obj, self)

        # Register the object
        self.ode_objects.append(obj)

class ODE(ODEComponent):
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

        ns = ns or {}

        # Reset constructed attribute
        self._constructed = False
        
        # Add Time object
        # FIXME: Add information about time unit dimensions and make
        # FIXME: it possible to have different time names
        time = Time("t", "ms")
        self._time = time
        self.ode_objects.append(time)

        # Namespace, which can be used to eval an expression
        self.ns = dict()

        # An ODEObjectList with all components
        # The components are always sorted wrt last expression added
        self.all_components = []

        # A dict with the present ode objects
        # NOTE: hashed by name so duplicated expressions are not stored
        self.present_ode_objects = dict(t=(self._time, self))
        self.duplicated_expressions = defaultdict(list)
        self.expression_dependencies = defaultdict(list)
        self.object_used_in = defaultdict(list)
        self.expanded_expressions = dict()

        self._constructed = True

    def register_ode_object(self, obj, comp):
        """
        Register an ODE object in the root ODEComponent
        """

        # Check for existing object in the ODE 
        duplication = self.present_ode_objects.get(obj.name)

        # If object with same name is already registered in the ode we
        # need to figure out what to do
        if duplication:
        
            dup_obj, dup_comp = duplication
        
            # If a parameter or Variable is registered using the same name
            # as the ODE it can be overwritten with the new object if the new object 
            #if isinstance(dub_obj, Parameter) and comp == self.root:

                # Remove global Parameter and 
            
            # If State, Parameter or DerivativeExpression we always raise an error
            if any(isinstance(oo, (State, Parameter, DerivativeExpression)) \
                   for oo in [dup_obj, obj]):
                error("Cannot register a {0}. A {1} with name '{2}' is "\
                      "already registered in this ODE.".format(\
                          type(obj).__name__, type(\
                              dup_obj).__name__, dup_obj.name))

            # Sanity check that both obj and dup_obj are Expressions
            assert all(isinstance(oo, (Expression)) for oo in [dup_obj, obj])

            # Get list of duplicated objects or an empy list
            dup_objects = self.duplicated_expressions[obj.name]
            if len(dup_objects) == 0:
                dup_objects.append(dup_obj)
            dup_objects.append(obj)

        # Update global information about ode object
        self.present_ode_objects[obj.name] = (obj, comp)
        self.ns[obj.name] = obj.sym

        # Register any Expression 
        self._register_expression(obj, comp)

    def _register_expression(self, obj, comp):

        # Do nothing if not Expression
        if not isinstance(obj, Expression):
            return
        
        # Append the name to all_components
        _bubble_append(self.all_components, comp.name)

        # Expand derivatives in the expressions
        for der_expr in obj.expr.atoms(sp.Derivative):

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
                continue

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

            # If we get a Derivative(expr, t) we issue an error
            if isinstance(expr_obj, Expression) and var_obj == self._time:
                error("All derivative expressions of registered expressions "\
                      "need to be expanded with respect to time. Use "\
                      "expr.diff(t) instead of Derivative(expr, t) ")
            
            # If the expr obj is an expression we perform the derivative and
            # store the expression
            if isinstance(expr_obj, Expression):
                comp.add_derivative(expr_obj, var_obj, \
                                    expr_obj.expr.diff(var_obj.sym))
                continue

            error("Should not reach here.")

        # Iterate over dependencies in the expression
        expression_subs = []
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
        
            # Collect intermediates to be used in substitutions below
            if isinstance(dep_obj, Expression):
                expression_subs.append((dep_obj.sym, \
                        self.expanded_expressions[dep_obj.name]))

        # Expand the Expression
        self.expanded_expressions[obj.name] = obj.expr.subs(\
                expression_subs)
        

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
