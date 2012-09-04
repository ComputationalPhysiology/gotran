__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2012 " + __author__
__date__ = "2012-02-22 -- 2012-09-04"
__license__  = "GNU LGPL Version 3.0 or later"

__all__ = ["ODE"]

# System imports
import inspect
from collections import OrderedDict, deque


# ModelParameter imports
from modelparameters.sympytools import ModelSymbol, sp, sp_namespace
from modelparameters.codegeneration import sympycode

# Gotran imports
from gotran2.common import error, check_arg, check_kwarg, scalars, listwrap
from gotran2.model.odeobjects import *

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
        self.clear()

    def add_state(self, name, init, comment=""):
        """
        Add a state to the ODE

        Arguments
        ---------
        name : str
            The name of the state variable
        init : scalar, ScalarParam
            The initial value of the state
        comment : str (optional)
        
            A comment which will follow the state
        Example:
        ========

        >>> ode = ODE("MyOde")
        >>> ode.add_state("e", 1)
        """
        
        # Create the state
        state = State(name, init, comment, self.name)
        
        # Register the state
        self._register_object(state)

        # Return the sympy version of the state
        return state.sym
        
    def add_parameter(self, name, init, comment=""):
        """
        Add a parameter to the ODE

        Arguments
        ---------
        name : str
            The name of the parameter
        init : scalar, ScalarParam
            The initial value of this parameter
        comment : str (optional)
            A comment which will follow the state
        
        Example:
        ========

        >>> ode = ODE("MyOde")
        >>> ode.add_parameter("c0", 5.0)
        """
        
        # Create the parameter
        parameter = Parameter(name, init, comment, self.name)
        
        # Register the parameter
        self._register_object(parameter)

        # Return the sympy version of the parameter
        return parameter.sym

    def add_variable(self, name, init, comment=""):
        """
        Add a variable to the ODE

        Arguments
        ---------
        name : str
            The name of the variables
        init : scalar, ScalarParam
            The initial value of this parameter
        comment : str (optional)
            A comment which will follow the state
        
        Example:
        ========

        >>> ode = ODE("MyOde")
        >>> ode.add_variable("c0", 5.0)
        """
        
        # Create the variable
        variable = Variable(name, init, comment, self.name)
        
        # Register the variable
        self._register_object(variable)

        # Return the sympy version of the variable
        return variable.sym

    def add_comment(self, comment_str):
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
                    value_error("expected derivatives to be a linearly weighted "\
                                "State variables.")

                # Grab ModelSymbol
                derivative = derivative.args[1]
                
            elif not isinstance(derivative, ModelSymbol):
                value_error("expected derivatives to be a linearly weighted "\
                            "State variables.")
            
            if not self.has_state(derivative):
                error("expected derivatives to be a declared state of this ODE")

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

        for atom in expanded_expr.atoms():
            if not isinstance(atom, (ModelSymbol, sp.Number, int, float)):
                type_error("A derivative must be an expressions of "\
                           "ModelSymbol or scalars")
                
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
        
    def iter_states(self):
        """
        Return an iterator over registered states
        """
        for state in self._all_objects.values():
            if isinstance(state, State):
                yield state

    def iter_field_states(self):
        """
        Return an iterator over registered field states
        """
        for state in self._all_objects.values():
            if isinstance(state, State) and state.is_field:
                yield state

    def iter_variables(self):
        """
        Return an iterator over registered variables
        """
        for variable in self._all_objects.values():
            if isinstance(variable, Variable):
                yield variable

    def iter_parameters(self):
        """
        Return an iterator over registered parameters
        """
        for parameter in self._all_objects.values():
            if isinstance(parameter, Parameter):
                yield parameter

    def has_state(self, state):
        """
        Return True if state is a registered state or field state
        """
        check_arg(state, (str, ModelSymbol, ODEObject))
        if isinstance(state, (str, ModelSymbol)):
            state = self.get_object(state)
            return isinstance(state, State)
        
        if not isinstance(state, State):
            return False
        
        return any(state == st for st in self.iter_states())
        
    def has_field_state(self, state):
        """
        Return True if state is a registered field state
        """
        check_arg(state, (str, ModelSymbol, ODEObject))
        if isinstance(state, (str, ModelSymbol)):
            state = self.get_object(state)
            return isinstance(state, State) and state.is_field 
        
        if not isinstance(state, State):
            return False
        
        return any(state == st for st in self.iter_field_states())
        
    def has_variable(self, variable):
        """
        Return True if variable is a registered variable
        """
        check_arg(variable, (str, ModelSymbol, ODEObject))
        if isinstance(variable, (str, ModelSymbol)):
            variable = self.get_object(variable)
            return isinstance(variable, Variable)
        
        if not isinstance(Variable):
            return False
        
        return any(variable == var for var in self.iter_variables())
        
    def has_parameter(self, param):
        """
        Return True if state is a registered parameter
        """
        check_arg(param, (str, ModelSymbol, ODEObject))
        if isinstance(param, (str, ModelSymbol)):
            param = self.get_object(param)
            return isinstance(param, Parameter)
        
        if not isinstance(param, Parameter):
            return False
        
        return any(param == par for par in self.iter_parameters())

    @property
    def name(self):
        return self._name

    @property
    def num_states(self):
        return len([s for s in self.iter_states()])
        
    @property
    def num_field_states(self):
        return len([s for s in self.iter_field_states()])
        
    @property
    def num_parameters(self):
        return len([s for s in self.iter_parameters()])
        
    @property
    def num_variables(self):
        return len([s for s in self.iter_variables()])

    @property
    def num_derivative_expr(self):
        return len(self._derivative_expr)
        
    @property
    def num_algebraic_expr(self):
        return len(self._algebraic_expr)

    @property
    def is_complete(self):
        """
        Check that the ODE is complete
        """
        states = [state for state in self.iter_states()]

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
        if len(self._algebraic_states) + len(self._derivative_states) != len(states):
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
                         "state is allowed per diff call: {0}".format(derivative))
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
        return len(self._all_objects) == 1

    def clear(self):
        """
        Clear any registered objects
        """

        # FIXME: Make this a dict of lists
        self._all_objects = OrderedDict()
        self._derivative_states = set() # FIXME: No need for a set here...
        self._algebraic_states = set()

        # Collection of intermediate stuff
        self._expansion_namespace = OrderedDict()
        self._intermediates = OrderedDict()
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
        self.add_variable("t", 0.0)
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
                error("Illeagal name '{0}'. It is already a registered {1}"\
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
            # Evaluate expression in expansion namespace. Remove the ode name prefix
            # in any included ModelSymbols

            self._expansion_namespace[name] = eval(
                sympycode(expr).replace(self.name+".",""), \
                self._expansion_namespace, {})
        except NameError, e:
            raise NameError("{0}. Registering an intermediate failed because of "\
                            "missmatch between \nthe ode's expansion namespace "\
                            "and the global namespace. If needed update the \n"\
                            "ode's namespace using, "\
                            "ode.update_expansion_namespace()".format(e))

    def _register_object(self, obj):
        """
        Register an object to the ODE
        """
        assert(isinstance(obj, ODEObject))
        
        # Register the object
        self._all_objects[obj.name] = obj

        # Make object available as an attribute
        setattr(self, obj.name, obj.sym)

        # Register symbol in the expansion namespace
        self._expansion_namespace[obj.name] = obj.sym

    def _sort_collected_objects(self):
        """
        Sort the collected object after alphabetic order
        """ 
        cmp_func = lambda a, b: cmp(a.name, b.name)
        self._states.sort(cmp_func)
        self._field_states.sort(cmp_func)
        self._parameters.sort(cmp_func)
        self._variables.sort(cmp_func)

    def __setattr__(self, name, value):
        if name[0] == "_":
            self.__dict__[name] = value
        elif name in self._all_objects:
            
            # Called when registering the ODEObject
            if hasattr(self, name):
                obj = self.get_object(attr)
                error("Illeagal name '{0}'. It is already a registered {1}"\
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
        
        # Sort all collected objects
        self._sort_collected_objects()
        other._sort_collected_objects()

        # Compare the list of objects
        for what in ["_states", "_field_states", "_parameters", "_variables"]:
            if getattr(self, what) != getattr(other, what):
                return False

        # Check equal differentiation
        # FIXME: Remove dependent
        for state in self.iter_states():
            if len(state.diff_expr) != len(other.get_object(state).diff_expr):
                return False
            for dependent, expr in state.diff_expr.items():
                if dependent not in other.get_object(state).diff_expr:
                    return False
                if other.get_object(state).diff_expr[dependent] != \
                   state.diff_expr[dependent]:
                    return False
        
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

