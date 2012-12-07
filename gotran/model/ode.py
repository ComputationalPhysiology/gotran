# Copyright (C) 2012 Johan Hake
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

__all__ = ["ODE"]

# System imports
import inspect
from collections import OrderedDict, deque


# ModelParameter imports
from modelparameters.sympytools import ModelSymbol, sp, sp_namespace
from modelparameters.codegeneration import sympycode

# Gotran imports
from gotran.common import type_error, value_error, error, check_arg, \
     check_kwarg, scalars, listwrap, info, debug
from gotran.model.odeobjects import *

class ODE(object):
    """
    Basic class for storying information of an ODE
    """
        
    def __init__(self, name):
        """
        Initialize an ODE
        
        Arguments
        ---------
        name : str
            The name of the ODE
        """
        check_arg(name, str, 0)

        # Initialize attributes
        self._name = name

        # Initialize all variables
        self._all_single_ode_objects = OrderedDict()
        self._states = ODEObjectList()
        self._field_states = ODEObjectList()
        self._parameters = ODEObjectList()
        self._field_parameters = ODEObjectList()
        self._variables = ODEObjectList()

        # FIXME: Move to list when we have a dedicated Intermediate class
        # FIXME: Change name to body?
        self._intermediates = ODEObjectList()
        self._monitored_intermediates = OrderedDict()

        # Add components
        self._default_component = ODEComponent(name, self)
        self._present_component = self._default_component
        self._components = OrderedDict()
        self._components[name] = self._default_component
        
        self.clear()

    def add_state(self, name, init, der_init=0.0, component=""):
        """
        Add a state to the ODE

        Arguments
        ---------
        name : str
            The name of the state variable
        init : scalar, ScalarParam
            The initial value of the state
        der_init : scalar, ScalarParam
            The initial value of the state derivative
        component : str (optional)
            Add state to a particular component
            
        Example:
        ========

        >>> ode = ODE("MyOde")
        >>> ode.add_state("e", 1)
        """

        # Create the state and derivative
        state = State(name, init, component, self.name)
        state_der = StateDerivative(state, der_init, component, self.name)
            
        self._register_object(state)
        self._register_object(state_der)
        
        # Append state specific information
        self._states.append(state)
        if state.is_field:
            self._field_states.append(state)

        # Return the sympy version of the state
        return state.sym
        
    def add_parameter(self, name, init, component=""):
        """
        Add a parameter to the ODE

        Arguments
        ---------
        name : str
            The name of the parameter
        init : scalar, ScalarParam
            The initial value of this parameter
        component : str (optional)
            Add state to a particular component
        
        Example:
        ========

        >>> ode = ODE("MyOde")
        >>> ode.add_parameter("c0", 5.0)
        """
        
        # Create the parameter
        parameter = Parameter(name, init, component, self.name)
        
        # Register the parameter
        self._register_object(parameter)
        self._parameters.append(parameter)
        if parameter.is_field:
            self._field_parameters.append(parameter)

        # Return the sympy version of the parameter
        return parameter.sym

    def add_variable(self, name, init, component=""):
        """
        Add a variable to the ODE

        Arguments
        ---------
        name : str
            The name of the variables
        init : scalar, ScalarParam
            The initial value of this parameter
        component : str (optional)
            Add state to a particular component
        
        Example:
        ========

        >>> ode = ODE("MyOde")
        >>> ode.add_variable("c0", 5.0)
        """
        
        # Create the variable
        variable = Variable(name, init, component, self.name)
        
        # Register the variable
        self._register_object(variable)
        self._variables.append(variable)

        # Return the sympy version of the variable
        return variable.sym

    def add_states(self, component="", **kwargs):
        """
        Add a number of states to the current ODE
    
        Example
        -------
    
        >>> ode = ODE("MyOde")
        >>> ode.states(e=0.0, g=1.0)
        """
    
        if not kwargs:
            error("expected at least one state")
        
        # Check values and create sympy Symbols
        self._add_entities(component, kwargs, "state")
    
    def add_parameters(self, component="", **kwargs):
        """
        Add a number of parameters to the current ODE
    
        Example
        -------
    
        >>> ode = ODE("MyOde")
        >>> ode.add_parameters(v_rest=-85.0,
                               v_peak=35.0,
                               time_constant=1.0)
        """
        
        if not kwargs:
            error("expected at least one state")
        
        # Check values and create sympy Symbols
        self._add_entities(component, kwargs, "parameter")
        
    def add_variables(self, component="", **kwargs):
        """
        Add a number of variables to the current ODE
    
        Example
        -------
    
        >>> ode = ODE("MyOde")
        >>> ode.add_variables(c_out=0.0, c_in=1.0)
        
        """
    
        if not kwargs:
            error("expected at least one variable")
        
        # Check values and create sympy Symbols
        self._add_entities(component, kwargs, "variable")

    def add_monitored(self, *args):
        """
        Add intermediate variables to be monitored

        Arguments
        ---------
        args : any number of intermediates
            Intermediates which will be monitored
        """

        for i, arg in enumerate(args):
            check_arg(arg, (str, ModelSymbol), i)
            obj = self.get_object(arg)

            if not isinstance(obj, Intermediate):
                error("Can only monitor indermediates. '{0}' is not an "\
                      "Intermediate.".format(obj.name))
            
            # Register the expanded monitored intermediate
            self._monitored_intermediates[name] = obj

    def add_intermediate(self, name, expr):
        """
        Add an Intermediate to present 
        """
        # FIXME: Add stuff...

    def _add_entities(self, component, kwargs, entity):
        """
        Help function for determine if each entity in the kwargs is unique
        and to check the type of the given default value
        """
        assert(entity in ["state", "parameter", "variable"])
    
        # Get caller frame
        # namespace = _get_load_namespace()
    
        # Get add method
        add = getattr(self, "add_{0}".format(entity))
        
        # Symbol and value dicts
        for name, value in sorted(kwargs.items()):
    
            # Add the symbol
            sym = add(name, value, component=component)

            # FIXME: Should we add this capability back?
            # Add symbol to caller frames namespace
            #try:
            #    debug("Adding {0} '{1}' to namespace".format(entity, name))
            #    if name in namespace:
            #        warning("Symbol with name: '{0}' already in namespace.".\
            #                format(name))
            #    namespace[name] = sym
            #except:
            #    error("Not able to add '{0}' to namespace".format(name))
    
    def add_comment(self, comment_str):
        """
        Add comment to ODE
        """
        check_arg(comment_str, str, context=ODE.add_comment)
        self._intermediates.append(\
            Comment(comment_str, self._present_component.name))

    def set_component(self, component):
        """
        Set present component
        """
        check_arg(component, str, 0, ODE.set_component)

        # Check if component is already registered
        comp = self._components.get(component)
        if comp is None:
            comp = ODEComponent(component, self)
            self._components[component] = comp
        self._present_component = comp

        # Update list of intermediates
        self._intermediates.append(comp)

    def get_object(self, name):
        """
        Return a registered object
        """

        check_arg(name, (str, ModelSymbol))
        if isinstance(name, ModelSymbol):
            name = name.name
        
        return self._all_single_ode_objects.get(name)

    def get_intermediate(self, name):
        """
        Return a registered intermediate
        """
        return self._intermediates.get(name)

    def diff(self, derivatives, expr, component=""):
        """
        Register an expression for state derivatives

        Arguments
        ---------
        derivatives : int, sympy.Basic
            A linear expression of StateDerivative symbols. Can also be an
            integer and then only 0, to add an algebraic expression
        expr : Sympy expression of ModelSymbols
            The derivative expression
        component : str (optional)
            The component will be determined automatically if the
            DerivativeExpression is an Algebraic expression
        """

        derivative_expression = DerivativeExpression(\
            derivatives, expr, self, component)
        
        
        # Store expressions
        if derivative_expression.is_algebraic:
            self._algebraic_expr.append(derivative_expression.expr)
            self._algebraic_expr_expanded.append(\
                derivative_expression.expanded_expr)
        else:
            self._derivative_expr.append((derivative_expression.states, \
                                          derivative_expression.expr))
            self._derivative_expr_expanded.append(\
                (derivative_expression.states, \
                 derivative_expression.expanded_expr))

        # Store derivative expression
        self._derivative_expressions.append(derivative_expression)

        # Store derivative states
        self._derivative_states.update(derivative_expression.states)

        # Register obj in component
        comp_name = derivative_expression.component
        
        comp = self._components.get(comp_name)
        if comp is None:
            error("Should never come here")
            comp = ODEComponent(comp_name, self)
            self._components[comp_name] = comp

        # Add object to component
        comp.append(derivative_expression)

    def get_derivative_expr(self, expanded=False):
        """
        Return the derivative expression
        """

        if expanded:
            return self._derivative_expr_expanded
        else:
            return self._derivative_expr

    def get_algebraic_expr(self, expanded=False):
        """
        Return the algebraic expression
        """
        if expanded:
            return self._algebraic_expr_expanded
        else:
            return self._algebraic_expr

    @property
    def states(self):
        """
        Return a list of all states 
        """
        return self._states

    @property
    def field_states(self):
        """
        Return a list of all field states 
        """
        return self._field_states

    @property
    def parameters(self):
        """
        Return a list of all parameters 
        """
        return self._parameters

    @property
    def field_parameters(self):
        """
        Return a list of all field parameters
        """
        return self._field_parameters

    @property
    def variables(self):
        """
        Return a list of all variables 
        """
        return self._variables

    @property
    def components(self):
        """
        Return a all components
        """
        return self._components

    @property
    def intermediates(self):
        """
        Return a all components
        """
        return self._intermediates

    @property
    def monitored_intermediates(self):
        """
        Return an dict over registered monitored intermediates
        """
        return _monitored_intermediates

    def has_state(self, state):
        """
        Return True if state is a registered state or field state
        """
        check_arg(state, (str, ModelSymbol, ODEObject))

        # Grab ODEObject if str or ModelSymbol is passed
        if isinstance(state, (str, ModelSymbol)):
            state = self.get_object(state)
            if state is None:
                return False
            
        return state in self._states
        
    def has_field_state(self, state):
        """
        Return True if state is a registered field state
        """
        check_arg(state, (str, ModelSymbol, ODEObject))

        # Grab ODEObject if str or ModelSymbol is passed
        if isinstance(state, (str, ModelSymbol)):
            state = self.get_object(state)
            if state is None:
                return False
            
        return state in self._field_states
        
    def has_parameter(self, parameter):
        """
        Return True if parameter is a registered parameter or field parameter
        """
        check_arg(parameter, (str, ModelSymbol, ODEObject))

        # Grab ODEObject if str or ModelSymbol is passed
        if isinstance(parameter, (str, ModelSymbol)):
            parameter = self.get_object(parameter)
            if parameter is None:
                return False

        return parameter in self._parameters
        
    def has_field_parameter(self, parameter):
        """
        Return True if parameter is a registered field parameter
        """
        check_arg(parameter, (str, ModelSymbol, ODEObject))

        # Grab ODEObject if str or ModelSymbol is passed
        if isinstance(parameter, (str, ModelSymbol)):
            parameter = self.get_object(parameter)
            if parameter is None:
                return False

        return parameter in self._field_parameters
        
    def has_variable(self, variable):
        """
        Return True if variable is a registered Variable
        """
        check_arg(variable, (str, ModelSymbol, ODEObject))

        # Grab ODEObject if str or ModelSymbol is passed
        if isinstance(variable, (str, ModelSymbol)):
            variable = self.get_object(variable)
            if variable is None:
                return False

        return variable in self._variables
        
    @property
    def name(self):
        return self._name

    @property
    def num_states(self):
        return len(self._states)
        
    @property
    def num_field_states(self):
        return len(self._field_states)
        
    @property
    def num_parameters(self):
        return len(self._parameters)
        
    @property
    def num_field_parameters(self):
        return len(self._field_parameters)
        
    @property
    def num_variables(self):
        return len(self._variables)

    @property
    def num_derivative_expr(self):
        return len(self._derivative_expr)
        
    @property
    def num_algebraic_expr(self):
        return len(self._algebraic_expr)

    @property
    def num_monitored_intermediates(self):
        """
        Return the number of monitored intermediates
        """
        return len(self._monitored_intermediates)

    @property
    def expansion_subs(self):
        return self._expansion_subs

    @property
    def is_complete(self):
        """
        Check that the ODE is complete
        """
        states = self._states

        if not states:
            return False

        if len(states) > self.num_derivative_expr + self.num_algebraic_expr:
            # FIXME: Need a better name instead of xpressions...
            info("The ODE is under determined. The number of States are more "\
                 "than the number of expressions.")
            return False

        if len(states) < self.num_derivative_expr + self.num_algebraic_expr:
            # FIXME: Need a better name instead of xpressions...
            info("The ODE is over determined. The number of States are less "\
                 "than the number of expressions.")
            return False
        
        # Grab algebraic states
        self._algebraic_states.update(states)
        self._algebraic_states.difference_update(self._derivative_states)

        # Concistancy check
        if len(self._algebraic_states) + len(self._derivative_states) \
               != len(states):
            info("The sum of the algebraic and derivative states need equal "\
                 "the total number of states.")
            return False

        # If we have the same number of derivative expressions as number of
        # states we need to check that we have one derivative of each state.
        # and sort the derivatives

        if self.num_derivative_expr == len(states):

            for derivative, expr in self._derivative_expr:
                if len(derivative) != 1:
                    # FIXME: Better error message
                    info("When no DAE expression is used only 1 differential "\
                         "state is allowed per diff call: {0}".format(\
                             derivative))
                    return False

            # Get a copy of the lists and prepare sorting
            derivative_expr = self._derivative_expr[:]
            derivative_expr_expanded = self._derivative_expr_expanded[:]
            derivative_expr_sorted = []
            derivative_expr_expanded_sorted = []

            for state in states:
                for ind, (derivative, expr) in enumerate(derivative_expr):
                    if derivative[0] == state.sym:
                        derivative_expr_sorted.append(derivative_expr.pop(ind))
                        derivative_expr_expanded_sorted.append(\
                            derivative_expr_expanded.pop(ind))
                        break

            # Store the sorted derivatives
            self._derivative_expr = derivative_expr_sorted
            self._derivative_expr_expanded = derivative_expr_expanded_sorted

        # Nothing more to check?
        return True

    @property
    def is_dae(self):
        return self.is_complete and len(self._algebraic_states) > 0

    @property
    def is_empty(self):
        """
        Returns True if the ODE is empty
        """
        # By default only t is a registered object
        return len(self._all_single_ode_objects) == 2

    def clear(self):
        """
        Clear any registered objects
        """

        # Delete stored attributes
        for name in self._all_single_ode_objects.keys():
            if name[0] == "_":
                continue
            delattr(self, name)

        for intermediate in self._intermediates:
            try:
                delattr(self, intermediate.name)
            except:
                pass
            
        self._all_single_ode_objects = OrderedDict()
        self._derivative_expressions = []
        self._derivative_states = set() # FIXME: No need for a set here...
        self._algebraic_states = set()

        # Collection of intermediate stuff
        self._expansion_subs = []
        self._intermediates = ODEObjectList()
        self._comment_num = 0
        self._duplicate_num = 0

        # Collect expressions (Expanded and intermediate kept)
        self._derivative_expr = []
        self._derivative_expr_expanded = []
        self._algebraic_expr = []
        self._algebraic_expr_expanded = []

        # Analytics (not sure we need these...)
        self._dependencies = {}
        self._linear_dependencies = {}

        # Add time as a variable
        self.add_variable("time", 0.0, self._default_component.value)
        self.add_variable("dt", 0.1, self._default_component.value)
        
    def _register_object(self, obj):
        """
        Register an ODEObject (only used for states, parameters, variables,
        and state_derivatives)
        """
        
        assert(isinstance(obj, (State, Parameter, Variable, StateDerivative)))

        # Check for existing object
        dup_obj = self._all_single_ode_objects.get(obj.name)
        if dup_obj is not None:
            error("A '{0}' with name '{1}' is already registered in this "\
                  "ODE.".format(dup_obj, obj.name))

        # Check that we have not started registering intermediates
        if (len(self._intermediates) + len(self._derivative_expressions)) > 0:
            error("Cannot register a {0} after an intermediate or derivative "\
                  "have been registered.".format(type(obj).__name__))

        # Register obj in component
        comp = self._components.get(obj.component)
        if comp is None:
            comp = ODEComponent(obj.component, self)
            self._components[obj.component] = comp

        # Update present component
        self._present_component = comp

        # Add object to component if not StateDerivative
        if not isinstance(obj, StateDerivative):
            comp.append(obj)

        # Register the object
        self._all_single_ode_objects[obj.name] = obj

        # Register the symbol of the Object as an attribute
        self.__dict__[obj.name] = obj.sym

    def _register_intermediate(self, name, expr):
        """
        Register an intermediate
        """

        # Check that we have not started registering derivatives
        if self._derivative_expressions:
            error("Cannot register an Intermediate after "\
                  "a DerivativeExpression has been registered.")

        # Create an intermediate in the present component
        intermediate = Intermediate(name, expr, self, \
                                    self._present_component.name)
        
        # Store the intermediate
        self._intermediates.append(intermediate)

        # Add to component
        self._present_component.append(intermediate)

        # Register symbol, overwrite any already excisting symbol
        self.__dict__[name] = intermediate.sym

        # Update expansion subs
        self._expansion_subs.append((intermediate.sym, 
                                     intermediate.expanded_expr))

    def __setattr__(self, name, value):
        """
        A magic function which will register intermediates and simpler
        derivative expressions
        """

        # If we are registering a protected attribute, just add it to
        # the dict
        if name[0] == "_":
            self.__dict__[name] = value
            return
        
        # Check if name is a registered SingleODEObject
        obj = self.get_object(name)

        # If the object is already registered
        if obj is not None:
            
            # If state derivative we call diff with the state derivative
            # symbol as first argument
            if isinstance(obj, StateDerivative):
                self.diff(obj.sym, value)
                return True

            error("{0} is already a registerd ODEObject, which cannot "\
                  "be overwritten.".format(name))

        # Assume that we are registering an intermediate
        if isinstance(value, sp.Basic) and any(isinstance(atom, ModelSymbol)\
                                               for atom in value.atoms()):
            self._register_intermediate(name, value)
            return True

        # If not registering Intermediate or doing a shorthand
        # diff expression we silently leave.
        

    def __eq__(self, other):
        """
        x.__eq__(y) <==> x==y
        """
        if not isinstance(other, ODE):
            return False

        if id(self) == id(other):
            return True
        
        # Compare all registered attributes
        subs = {}
        for what, item in self._all_single_ode_objects.items():
            if not hasattr(other, what):
                return False
            if getattr(self, what) != getattr(other, what):
                return False
            subs[getattr(self, what)] = getattr(other, what)

        for what, item in self._intermediates.items():
            if what[0] == "_":
                continue
            if not hasattr(other, what):
                return False
            if getattr(self, what) != getattr(other, what):
                return False
            subs[getattr(self, what)] = getattr(other, what)

        # FIXME: Fix comparison of expressions
        #print "self:", self._derivative_expr
        #print "other:", other._derivative_expr

        #for derivatives, expr in self._derivative_expr:
        #    if (derivatives, expr.subs(subs)) not in other._derivative_expr:
        #        print "Nope..."
        #        return False
        #
        #for derivatives, expr in self._derivative_expr_expanded:
        #    if (derivatives, expr.subs(subs)) not in \
        #               other._derivative_expr_expanded:
        #        print "Nopeidope..."
        #        return False

        return True

    def __str__(self):
        """
        x.__str__() <==> str(x)
        """
        return self.name
        
    def __repr__(self):
        """
        x.__repr__() <==> repr(x)
        """
        return "{}('{}')".format(self.__class__.__name__, self.name)


