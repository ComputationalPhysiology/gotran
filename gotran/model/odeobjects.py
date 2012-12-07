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

__all__ = ["ODEObject", "SingleODEObject", "Parameter", "State", \
           "StateDerivative", "Variable", "Expression", "ODEObjectList", \
           "SingleODEObject", "ODEComponent", "Intermediate", "Comment",\
           "DerivativeExpression",
           ]

#           "DiscreteVariable", "ContinuousVariable", ]

# System imports
import numpy as np
from collections import OrderedDict

# ModelParameters imports
from modelparameters.sympytools import sp, ModelSymbol, \
     iter_symbol_params_from_expr
from modelparameters.parameters import *

from gotran.common import error, check_arg, scalars, debug, DEBUG, \
     get_log_level, Timer

class ODEObject(object):
    """
    Base container class for all ODEObjects
    """
    def __init__(self, name, value, component="", ode_name=""):
        """
        Create ODEObject instance

        Arguments
        ---------
        name : str
            The name of the ODEObject
        value : scalar, ScalarParam, np.ndarray, sp. Basic, str
            The value of this ODEObject
        component : str (optional)
            A component about the ODEObject
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """

        check_arg(name, str, 0, ODEObject)
        check_arg(value, scalars + (ScalarParam, list, np.ndarray, sp.Basic, str), \
                  1, ODEObject)
        check_arg(component, str, 2, ODEObject)
        check_arg(ode_name, str, 3, ODEObject)

        # Check for underscore in name
        if len(name) > 0 and name[0] == "_":
            error("No ODEObject names can start with an underscore: "\
                  "'{0}'".format(name))

        if isinstance(value, ScalarParam):

            # If Param already has a symbol
            if value.sym != dummy_sym:

                # Re-create one without a name
                value = eval(repr(value).split(", name")[0]+")")

        elif isinstance(value, scalars):
            value = ScalarParam(value)
        
        elif isinstance(value, (list, np.ndarray)):
            value = ArrayParam(np.fromiter(value, dtype=np.float_))
        elif isinstance(value, str):
            value = ConstParam(value)
        else:
            value = SlaveParam(value)

        # Debug
        if get_log_level() <= DEBUG:
            if isinstance(value, SlaveParam):
                debug("{0}: {1} {2:.3f}".format(name, value.expr, value.value))
            else:
                debug("{0}: {1}".format(name, value.value))
            
        # Create a symname based on the name of the ODE
        if ode_name:
            value.name = name, "{0}.{1}".format(ode_name, name)
        else:
            value.name = name

        # Store the Param
        self._param = value 

        # Store field
        self._field = isinstance(value, ArrayParam)
        self._component = component
        self._ode_name = ode_name

    @property
    def is_field(self):
        return self._field

    @property
    def sym(self):
        return self._param.sym

    @property
    def name(self):
        return self._param.name

    @property
    def param(self):
        return self._param

    @property
    def component(self):
        return self._component

    def __eq__(self, other):
        """
        x.__eq__(y) <==> x==y
        """
        
        if not isinstance(other, type(self)):
            return False
        
        # FIXME: Should this be more restrictive? Only comparing ODEObjects,
        # FIXME: and then comparing name and component?
        # FIXME: Yes, might be some side effects though...
        # FIXME: Need to do change when things are stable
        return self.name == str(other)

    def __ne__(self, other):
        """
        x.__neq__(y) <==> x==y
        """
        
        if not isinstance(other, type(self)):
            return True
        
        # FIXME: Should this be more restrictive? Only comparing ODEObjects,
        # FIXME: and then comparing name and component?
        # FIXME: Yes, might be some side effects though...
        # FIXME: Need to do change when things are stable
        return self.name != str(other)

    def __str__(self):
        """
        x.__str__() <==> str(x)
        """
        return self.name

    def __repr__(self):
        """
        x.__repr__() <==> repr(x)
        """
        return "{0}({1})".format(self.__class__.__name__, self._args_str())

    def _args_str(self):
        """
        Return a formated str of __init__ arguments
        """
        return "'{0}', {1}{2}{3}".format(\
            self.name, repr(self._param.getvalue()),
            ", component='{0}'".format(self._component) \
            if self._component else "",
            ", ode_name='{0}'".format(self._ode_name) \
            if self._ode_name else "",)

class SingleODEObject(ODEObject):
    """
    A class for all ODE objects which are not compound
    """
    
    def __init__(self, name, init, component="", ode_name=""):
        """
        Create ODEObject instance

        Arguments
        ---------
        name : str
            The name of the ODEObject
        init : scalar, ScalarParam, np.ndarray
            The init value of this ODEObject
        component : str (optional)
            A component about the ODEObject
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """

        # Init super class
        super(SingleODEObject, self).__init__(name, init, component, ode_name)

    @property
    def init(self):
        return self._param.getvalue()

    @init.setter
    def init(self, value):
        self._param.setvalue(value)

class State(SingleODEObject):
    """
    Container class for a State variable
    """
    def __init__(self, name, init, component="", ode_name=""):
        """
        Create a state variable with an assosciated initial value

        Arguments
        ---------
        name : str
            The name of the State
        init : scalar, ScalarParam
            The initial value of this state
        component : str (optional)
            A component for which the State should be associated with.
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """
        
        # Call super class
        super(State, self).__init__(name, init, component, ode_name)

        self.derivative = None

        # Add previous value symbol
        if ode_name:
            self.sym_0 = ModelSymbol("{0}_0".format(name), \
                                     "{0}.{1}_0".format(ode_name, name))
        else:
            self.sym_0 = ModelSymbol("{0}_0".format(name))
    
class StateDerivative(SingleODEObject):
    """
    Container class for a StateDerivative variable
    """
    def __init__(self, state, init=0.0, component="", ode_name=""):
        """
        Create a state derivative variable with an assosciated initial value

        Arguments
        ---------
        state : State
            The State
        init : scalar, ScalarParam
            The initial value of this state derivative
        component : str (optional)
            A component for which the State should be associated with.
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """

        check_arg(state, State)

        # Call super class
        super(StateDerivative, self).__init__("d{0}_dt".format(state.name), \
                                              init, component, ode_name)
        
        self.state = state
    
class Parameter(SingleODEObject):
    """
    Container class for a Parameter
    """
    def __init__(self, name, init, component="", ode_name=""):
        """
        Create a Parameter with an assosciated initial value

        Arguments
        ---------
        name : str
            The name of the State
        init : scalar, ScalarParam
            The initial value of this parameter
        component : str (optional)
            A component for which the Parameter should be associated with.
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """
        
        # Call super class
        super(Parameter, self).__init__(name, init, component, ode_name)

class Variable(SingleODEObject):
    """
    Container class for a Variable
    """
    def __init__(self, name, init, component="", ode_name=""):
        """
        Create a variable with an assosciated initial value

        Arguments
        ---------
        name : str
            The name of the variable
        init : scalar
            The initial value of this variable
        component : str (optional)
            A component for which the Variable should be associated with.
        ode_name : str (optional)
            The name of the ODE the ODEObject belongs to
        """
        
        # Call super class
        super(Variable, self).__init__(name, init, component, ode_name)

        # Add previous value symbol
        self.sym_0 = ModelSymbol("{0}_0".format(name), \
                                 "{0}.{1}_0".format(ode_name, name))


class Expression(ODEObject):
    """
    class for all expressions such as intermediates and diff 
    """
    def __init__(self, name, expr, ode, component=""):
        """
        Create an Exression with an assosciated name

        Arguments
        ---------
        name : str
            The name of the Expression
        expr : sympy.Basic
            The expression 
        ode : ODE
            The ODE which the expression is declared within
        component : str (optional)
            A component for which the Expression should be associated with.
        """

        # Check arguments
        from gotran.model.ode import ODE
        check_arg(expr, sp.Basic, 0, Expression)
        check_arg(ode, ODE, 1, Expression)
        
        if not any(isinstance(atom, (ModelSymbol, sp.Number)) \
                   for atom in expr.atoms()):
            error("expected the expression to contain at least one "\
                  "ModelSymbol or Number.")

        # Check that we are not using a DerivativeExpressions in expression
        for sym in iter_symbol_params_from_expr(expr):
            dep_obj = ode.get_object(sym) or ode._intermediates.get(sym)
            if dep_obj is None:
                error("The symbol '{0}' is not declared within the '{1}' "\
                      "ODE.".format(sym, ode.name))
            if isinstance(dep_obj, (StateDerivative, DerivativeExpression)):
                error("An expression cannot include a StateDerivative or "\
                      "DerivativeExpression")

        # Call super class with expression as the "value"
        super(Expression, self).__init__(name, expr, component, ode.name)

        # Create and store expanded expression
        timer = Timer("subs")
        self._expanded_expr = expr.subs(ode.expansion_subs)

    @property
    def value(self):
        return self._param.getvalue()

    @value.setter
    def value(self, value):
        self._param.setvalue(value)

    @property
    def expr(self):
        """
        Return the stored expression
        """
        return self._param.expr

    @property
    def expanded_expr(self):
        """
        Return the stored expression
        """
        return self._expanded_expr

class DerivativeExpression(Expression):
    """
    class for all derivative expressions
    """
    def __init__(self, derivatives, expr, ode, component=""):
        """
        Create an derivative or algebraic expression

        Arguments
        ---------
        derivatives : int, sympy.Basic
            A linear expression of StateDerivative symbols. Can also be an
            integer and then only 0, to add an algebraic expression
        expr : sympy.Basic
            The expression 
        ode : ODE
            The ODE which the derivative expression is declared within
        component : str (optional)
            The component will be determined automatically if the
            DerivativeExpression is an Algebraic expression
        """

        check_arg(derivatives, (sp.Basic, ModelSymbol, int), 0)

        error_str = "expected a linear combination of state derivatives "\
                    "as the derivative argument."

        def check_single_model_sym(sym):
            obj = ode.get_object(derivatives)
            if obj is None or not isinstance(obj, StateDerivative):
                error(error_str)
        
        def check_mul(mul):
            """
            Help function to check a mul operator in the derivative expression
            """
            derivative = None
            for arg in mul.args:
                if isinstance(arg, ModelSymbol):
                    obj = ode.get_object(arg)

                    if isinstance(obj, StateDerivative):

                        # Check that we have no registered Derivative 
                        if derivative is not None:
                            error(error_str)

                        # Save StateDerivative
                        derivative = obj
                        
                    elif isinstance(obj, Parameter):
                        # Parameters are fine
                        pass
                    else:
                        error(error_str)
                        
                elif arg.is_number:
                    # Numbers are fine
                    pass
                else:
                    error(error_str)

            # Check that we got one StateDerivative
            if derivative is None:
                error(error_str)

            # Return StateDerivative
            return derivative

        # Start checking derivatives

        stripped_derivatives = []

        # If an int we expect that to be Zero
        if isinstance(derivatives, int):
            if derivatives != 0:
                type_error("expected either an expression of derivatives or 0 "
                           "as the derivative arguments")
            name = "algebra"
  
        # If an expression of derivatives
        elif isinstance(derivatives, sp.Basic):

            # If single ModelSymbol we expect a DerivativeExpression
            if isinstance(derivatives, ModelSymbol):
                check_single_model_sym(derivatives)
                stripped_derivatives.append(ode.get_object(derivatives))

            # If derivatives is a linear combination of derivatives
            elif isinstance(derivatives, sp.Add):
                for arg in derivatives.args:
                    if isinstance(arg, ModelSymbol):
                        check_single_model_sym(arg)
                    elif isinstance(arg, sp.Mul):
                        stripped_derivatives.append(check_mul(arg))

            # If derivatives is a constantly weighted single derivative
            elif isinstance(derivatives, sp.Mul):
                derivative = check_mul(derivatives)
                warning("Got constant weighted single derivative. Divide "\
                        "whole expression with constant part.")
                constant_part = derivatives/derivative.sym
                expr = expr/constant_part
                derivatives = derivative.sym
                stripped_derivatives.append(derivative)
                
            else:
                type_error("expected a linear combination of "\
                           "derivatives as the derivative argument.")

            # Create name based on derivatives
            name = str(derivatives)

        # Store stripped_derivatives
        self._derivatives = stripped_derivatives

        # Check that all derivative states belong to the same component
        if len(stripped_derivatives) == 1:
            component = stripped_derivatives[0].component
        elif len(stripped_derivatives) > 1:
            component = stripped_derivatives[0].component
            if not all(component == der.component \
                       for der in stripped_derivatives[1:]):
                error("Expected all derivative expressions to belong to "\
                      "the same component")
        else:
            # No derivative expression, use component passed to
            # constructor
            pass
        
        # Call super class with expression as the "init" value
        super(DerivativeExpression, self).__init__(name, expr, ode, component)

    @property
    def num_derivatives(self):
        """
        Return the number of derivatives
        """
        return len(self._derivatives)

    @property
    def states(self):
        """
        Return the derivative states
        """
        return [der.state for der in self._derivatives]

    @property
    def is_algebraic(self):
        return not bool(self._derivatives)
    
class Intermediate(Expression):
    """
    class for all Intermediates 
    """
    def __init__(self, name, expr, ode, component=""):
        """
        Create an Intermediate with an assosciated name

        Arguments
        ---------
        name : str
            The name of the Intermediate
        expr : sympy.Basic
            The expression 
        expanded_expr : sympy.Basic
            The expanded verision of the intermediate 
        component : str (optional)
            A component for which the Intermediate should be associated with.
        """

        # Call super class with expression as the "init" value
        super(Intermediate, self).__init__(name, expr, ode, component)

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
        super(Comment, self).__init__(comment, "", component)

    @property
    def value(self):
        """
        Return the value of the Comment, which is the stored comment
        """
        return self._param.getvalue()
    
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
        
        check_arg(obj, ODEObject, 0, ODEComponent.append)

        assert(obj.component == self.name)

        # If SingleODEObject we need to check that no Expressions has been added
        if isinstance(obj, SingleODEObject):

            if (len(self.intermediates) + len(self.derivatives))>0:
                error("Cannot register a {0} to '{1}' after an expression"\
                      " has been register.".format(\
                          type(obj).__name__, self.name, ))

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
            
            # We have an expression and we need to figure out dependencies
            for sym in iter_symbol_params_from_expr(obj.expr):
                dep_obj = self._ode.get_object(sym) or \
                          self._ode.get_intermediate(sym)
                assert(dep_obj)
                assert(not isinstance(dep_obj, DerivativeExpression))
                self.external_object_dep.add(dep_obj)
                self.external_component_dep.add(dep_obj.component)

            if isinstance(obj, Intermediate):
                self.intermediates.append(obj)
            elif isinstance(obj, DerivativeExpression):
                self.derivatives.append(obj)
            else:
                error("Not recognised Expression: {0}".format(\
                    type(obj).__name__))
                
        elif isinstance(object, Comment):
            self.intermediates.append(obj)

        else:
            error("Not recognised ODEObject: {0}".format(\
                type(obj).__name__))

    @property
    def value(self):
        """
        Return the value of the Component, which is the str representation of it
        """
        return self._param.getvalue()
    
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
            return any(item == obj.sym for obj in self)
        elif (item, ODEObject):
            return super(ODEObjectList, self).__contains__(item)
        return False

    def count(self, item):
        if isinstance(item, str):
            return sum(item == obj.name for obj in self)
        elif isinstance(item, ModelSymbol):
            return sum(item == obj.sym for obj in self)
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
                if item == obj.sym:
                    return ind
        elif (item, ODEObject):
            for ind, obj in enumerate(self):
                if item == obj:
                    return ind
        raise ValueError("Item '{0}' not part of this ODEObjectList.".format(str(item)))

    def sort(self):
        error("Cannot alter ODEObjectList, other than adding ODEObjects.")

    def pop(self, item):
        error("Cannot alter ODEObjectList, other than adding ODEObjects.")

    def remove(self, item):
        error("Cannot alter ODEObjectList, other than adding ODEObjects.")

    def reverse(self, item):
        error("Cannot alter ODEObjectList, other than adding ODEObjects.")

    
