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
        self._all_objects = OrderedDict()
        self._states = []
        self._field_states = []
        self._parameters = []
        self._field_parameters = []
        self._variables = []

        # FIXME: Move to list when we have a dedicated Intermediate class
        self._intermediates = OrderedDict()
        self._monitored_intermediates = OrderedDict()
        
        self.clear()

    def add_state(self, name, init, component=""):
        """
        Add a state to the ODE

        Arguments
        ---------
        name : str
            The name of the state variable
        init : scalar, ScalarParam
            The initial value of the state
        component : str (optional)
            Add state to a particular component
            
        Example:
        ========

        >>> ode = ODE("MyOde")
        >>> ode.add_state("e", 1)
        """
        
        # Create the state
        state = State(name, init, component, self.name)
        
        # Register the state
        self._register_object(state)
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
            name = arg.name

            if name not in self._intermediates:
                error("Intermediate '{0}' is not a registered intermediate "\
                      "in this ODE.".format(name))

            assert(name in self._expansion_namespace)
            
            # Register the expanded monitored intermediate
            self._monitored_intermediates[name] = self._expansion_namespace[name]
            
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
            sym = add(name, value, component)

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
        self._intermediates["_comment_" + str(self._comment_num)] = comment_str
        self._comment_num += 1

    def get_object(self, name):
        """
        Return a registered object
        """

        check_arg(name, (str, ModelSymbol))
        if isinstance(name, ModelSymbol):
            name = name.name
        
        return self._all_objects.get(name)

    def diff(self, derivatives, expr):
        """
        Register an expression for state derivatives

        Arguments
        ---------
        derivatives : State, list of States or 0
            If derivatives is a single state then it is interpreted as an ODE
            If a list of states (with possible scalar weights) or 0 is
            given, it is interpreted as a DAE expression.
        expr : Sympy expression of ModelSymbols
            The derivative expression
            
        """
        check_arg(derivatives, (ModelSymbol, list, int), 0)
        
        if isinstance(derivatives, int) and derivatives != 0:
            type_error("expected either a State, a list of States or 0 "
                       "as the states arguments")

        derivatives = listwrap(derivatives or [])
        stripped_derivatives = []
        
        for derivative in derivatives:
            if isinstance(derivative, sp.Mul):
                if len(derivative.args) != 2 or \
                       not (derivative.args[0].is_number and \
                            isinstance(derivative.args[1]), ModelSymbol):
                    value_error("expected derivatives to be a linearly "\
                                "weighted State variables.")

                # Grab ModelSymbol
                derivative = derivative.args[1]
                
            elif not isinstance(derivative, ModelSymbol):
                value_error("expected derivatives to be a linearly weighted "\
                            "State variables.")
            
            if not self.has_state(derivative):
                error("expected derivatives to be a declared state "\
                      "of this ODE")

            # Register this state as used
            state = self.get_object(derivative)
            if state in self._derivative_states:
                error("A derivative for state '{0}' is already registered.")
                
            self._derivative_states.add(state)
            stripped_derivatives.append(state)

        # Register the derivatives
        check_arg(expr, (sp.Basic, scalars), 1)
        expanded_expr = eval(sympycode(expr).replace(self.name+".",""),\
                             self._expansion_namespace, {})

        expanded_expr = sp.sympify(expanded_expr)
        for atom in expanded_expr.atoms():
            # FIXME: Include Dummy to prevent bailout for Piecewise in sympy <= 0.7.2
            if not isinstance(atom, (ModelSymbol, sp.NumberSymbol, \
                                     sp.Number, int, float, sp.Dummy)):
                type_error("A derivative must be an expressions of "\
                           "ModelSymbol or scalars, got {0} which is "\
                           "a {1}.".format(atom, atom.__class__.__name__))
                
            if not isinstance(atom, ModelSymbol):
                continue

            # Get corresponding ODEObject
            sym = self.get_object(atom)

            if sym is None:
                error("ODEObject '{0}' is not registered in the "\
                      "'{1}' ODE".format(atom, self))

            # If a State
            if self.has_state(sym):

                # Check dependencies on other states
                # FIXME: What do we use this for...
                for derivative in stripped_derivatives:
                    if derivative not in self._dependencies:
                        self._dependencies[derivative] = set()
                    self._dependencies[derivative].add(sym)
                if len(stripped_derivatives) == 1 and \
                       atom not in expanded_expr.diff(atom).atoms():
                    if derivative not in self._linear_dependencies:
                        self._linear_dependencies[derivative] = set()
                    self._linear_dependencies[derivative].add(sym)

        # Store expressions
        # No derivatives (algebraic)
        if not derivatives:
            self._algebraic_expr.append(expr)
            self._algebraic_expr_expanded.append(expanded_expr)
        else:
            self._derivative_expr.append((derivatives, expr))
            self._derivative_expr_expanded.append((derivatives, \
                                                   expanded_expr))

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

    def iter_monitored_intermediates(self):
        """
        Return an iterator over registered monitored intermediates
        """
        for name, intermediate in self._monitored_intermediates.items():
            yield name, intermediate

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
        return len(self._all_objects) == 2

    def clear(self):
        """
        Clear any registered objects
        """

        # Delete stored attributes
        for name in self._all_objects.keys() + self._intermediates.keys():
            if name[0] == "_":
                continue
            delattr(self, name)
        
        self._all_objects = OrderedDict()
        self._derivative_states = set() # FIXME: No need for a set here...
        self._algebraic_states = set()

        # Collection of intermediate stuff
        self._expansion_namespace = OrderedDict()
        self._intermediates.clear()
        self._intermediates_duplicates = {} # Will be populated with deque
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
        self.add_variable("time", 0.0)
        self.add_variable("dt", 0.1)
        
        # Populate expansion namespace with sympy namespace
        self._expansion_namespace.update(sp_namespace)

    def update_expansion_namespace(self, **kwargs):
        """
        Update the namespace all intermediates gets expanded into.
        """
        self._expansion_namespace.update(kwargs)
        
    def _register_intermediate(self, name, expr):
        """
        Register an intermediate
        """

        # Check if attribute already exist
        if hasattr(self, name):

            # Check that attribute is not a state, variable or parameter
            attr = getattr(self, name)

            # If ModelSymbol 
            if not isinstance(attr, ModelSymbol):
                error("Illeagal name '{0}'. It is already an attribute "\
                      "of '{1}'".format(name, self.name))

            # If name is a ModelSymbol and not in intermediates it
            # state, variable or parameter
            # FIXME: Should not be nessesary to check...
            if name not in self._intermediates:
                obj = self.get_object(attr)
                assert(obj)
                error("Illeagal name '{0}'. It is already a registered {1} "\
                      "of '{2}'".format(name, obj.__class__.__name__, \
                                        self.name))

            # Register intermediate duplicate
            if name not in self._intermediates_duplicates:
                self._intermediates_duplicates[name] = deque()
            self._intermediates_duplicates[name].append(expr)
            self._intermediates["_duplicate_{0}".format(\
                self._duplicate_num)] = name
            self._duplicate_num += 1

        else:
            self._intermediates[name] = expr
            #print name, "=", expr
            sym = ModelSymbol(name, "{0}.{1}".format(self.name, name))
            setattr(self, name, sym)

        # Update namespace
        try:
            # Evaluate expression in expansion namespace. Remove the ode
            # name prefix in any included ModelSymbols

            self._expansion_namespace[name] = eval(
                sympycode(expr).replace(self.name+".",""), \
                self._expansion_namespace, {})
        except NameError, e:
            raise NameError("{0}. Registering an intermediate failed because"\
                            " of missmatch between \nthe ode's expansion "\
                            "namespace and the global namespace. If needed "\
                            "update the \node's namespace using, "\
                            "ode.update_expansion_namespace()".format(e))

    def _register_object(self, obj):
        """
        Register an object to the ODE
        """
        assert(isinstance(obj, ODEObject))
        
        # Register the object
        # FIXME: Add possibilities to add duplicates of an Intermediate
        if obj.name in self._all_objects:
            obj = self._all_objects[obj.name]
            error("Illeagal name '{0}'. It is already a registered {1} "\
                  "of '{2}'".format(obj.name, obj.__class__.__name__, \
                                    self.name))
        self._all_objects[obj.name] = obj

        # Make object available as an attribute
        setattr(self, obj.name, obj.sym)

        # FIXME: Should we add the name of the ode pointing to self, in
        # FIXME: the expansion namespace, making it possible to access
        # FIXME: names through attributes?
        # Register symbol in the expansion namespace
        self._expansion_namespace[obj.name] = obj.sym

    def __setattr__(self, name, value):
        if name[0] == "_":
            self.__dict__[name] = value
        elif name in self._all_objects:
            
            # Called when registering the ODEObject
            if hasattr(self, name):
                obj = self.get_object(name)
                error("Illeagal name '{0}'. It is already a registered {1} "\
                      "of '{2}'".format(name, obj.__class__.__name__, \
                                        self.name))
            self.__dict__[name] = value
        elif isinstance(value, ModelSymbol) and value.name == name:
            self.__dict__[name] = value
        else:
            self._register_intermediate(name, value)
            
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
        for what, item in self._all_objects.items():
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


