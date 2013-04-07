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
import re
from collections import OrderedDict, deque

# ModelParameter imports
from modelparameters.sympytools import ModelSymbol, sp, sp_namespace
from modelparameters.codegeneration import sympycode
from modelparameters.parameters import ScalarParam
from modelparameters.sympytools import iter_symbol_params_from_expr

# Gotran imports
from gotran.common import type_error, value_error, error, check_arg, \
     check_kwarg, scalars, listwrap, info, debug, Timer
from gotran.model.odeobjects import *
from gotran.model.odecomponents import *
from gotran.model.expressions import *

_derivative_name_template = re.compile("\Ad([a-zA-Z]\w*)_dt\Z")

class ODE(object):
    """
    Basic class for storying information of an ODE
    """
        
    def __init__(self, name, return_namespace=False):
        """
        Initialize an ODE
        
        Arguments
        ---------
        name : str
            The name of the ODE
        return_namespace : bool (optional)
            Several methods will return a namespace dict populated with symbols
            added to the ODE if return_namespace is set to True.
        """
        check_arg(name, str, 0)

        self._return_namespace = return_namespace

        # Initialize attributes
        self._name = name.strip().replace(" ", "_")

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
        self._intermediate_duplicates = set()
        self._monitored_intermediates = OrderedDict()
        self._comments = ODEObjectList()
        self._markov_models = ODEObjectList()

        # Add components
        self._default_component = ODEComponent(name, self)
        self._present_component = self._default_component
        self._components = OrderedDict()
        self._components[name] = self._default_component
        self._components_set = []

        self.clear()

    def add_state(self, name, init, der_init=0.0, component="", slaved=False, \
                  replace_level="none"):
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
        slaved : bool
            If True the creation and differentiation is controlled by
            other entity, like a Markov model.
        replace_level : str (optional)
            If not "none", replace can be either "global_params" or "any",
            meaning that either global or any object can be replaced by
            the added intermediate
            
        Example:
        ========

        >>> ode = ODE("MyOde")
        >>> ode.add_state("e", 1)
        """

        timer = Timer("Add state")

        # Create the state and derivative
        component = component or self.name
        state = State(name, init, component, self.name, slaved)
        state_der = StateDerivative(state, der_init, component, self.name)
        
        state.derivative = state_der
        
        self._register_object(state, replace_level)

        # FIXME: Remove register object when registering the state
        self._register_object(state_der)
        
        # Append state specific information
        self._states.append(state)
        if state.is_field:
            self._field_states.append(state)

        # Return the sympy version of the state
        return state.sym
        
    def add_parameter(self, name, init, component="", slaved=False, \
                      replace_level="none"):
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
        slaved : bool
            If True the creation and differentiation is controlled by
            other entity, like a Markov model.
        replace_level : str (optional)
            If not "none", replace can be either "global_params" or "any",
            meaning that either global or any object can be replaced by
            the added intermediate
            
        Example:
        ========

        >>> ode = ODE("MyOde")
        >>> ode.add_parameter("c0", 5.0)
        """
        
        timer = Timer("Add parameter")
        
        # Create the parameter
        component = component or self.name
        parameter = Parameter(name, init, component, self.name)
        
        # Register the parameter
        self._register_object(parameter, replace_level)
        self._parameters.append(parameter)
        if parameter.is_field:
            self._field_parameters.append(parameter)

        # Return the sympy version of the parameter
        return parameter.sym

    def add_variable(self, name, init, component="", slaved=False, \
                     replace_level="none"):
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
        slaved : bool
            If True the creation and differentiation is controlled by
            other entity, like a Markov model.
        replace_level : str (optional)
            If not "none", replace can be either "global_params" or "any",
            meaning that either global or any object can be replaced by
            the added intermediate
            
        Example:
        ========

        >>> ode = ODE("MyOde")
        >>> ode.add_variable("c0", 5.0)
        """
        
        timer = Timer("Add variable")

        # Create the variable
        component = component or self.name
        variable = Variable(name, init, component, self.name)
        
        # Register the variable
        self._register_object(variable, replace_level)
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
        return self._add_entities(component, sorted(kwargs.items()), "state")
    
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
        return self._add_entities(component, sorted(kwargs.items()), "parameter")
        
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
        return self._add_entities(component, sorted(kwargs.items()), "variable")

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

    def add_markov_model(self, name, component="", *args, **kwargs):
        """        
        Initialize a Markov model

        Arguments
        ---------
        name : str
            Name of Markov model
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

        # Create and store the markov model
        mm = MarkovModel(name, self, component=component, *args, **kwargs)
        
        self._markov_models.append(mm)

        # Register Markov model
        self._register_object(mm)

        # Return the markov model
        return mm
        
    def add_intermediate(self, name, expr, component=None, slaved=False, \
                         replace_level="global_params"):
        """
        Register an intermediate

        Arguments
        ---------
        name : str
            The name of the Intermediate
        expr : sympy.Basic
            The expression
        component : str (optional)
            A component name. If not given the present component will be used.
        slaved : bool
            If True the creation and differentiation is controlled by
            other entity, like a Markov model.
        replace_level : str (optional)
            If not "none", replace can be either "global_params" or "any",
            meaning that either global or any object can be replaced by
            the added intermediate
        """

        timer = Timer("Add intermediate")

        component_name = component if component else \
                         self._present_component.name
        
        # Create an intermediate in the present component
        intermediate = Intermediate(name, expr, self, component_name, slaved)

        check_kwarg(replace_level, "replace_level", str)

        # Check for existing object
        dup_obj = self._all_single_ode_objects.get(name)

        # If the object already exists and is a StateDerivative
        if dup_obj is not None and isinstance(dup_obj, StateDerivative):
            self.diff(dup_obj.sym, expr)
            return

        # If object already exists
        if dup_obj and not self._remove_duplicate(dup_obj, replace_level):
            error("Cannot register a {0}. A {1} with name '{2}' is "\
                  "already registered in this ODE.".format(\
                      type(intermediate).__name__, type(dup_obj).__name__, dup_obj.name))
        
        # Check for reserved wording of StateDerivatives
        if re.search(_derivative_name_template, intermediate.name):
            error("The pattern d{{name}}_dt is reserved for derivatives. "
                  "However {0} is not a state derivative.".format(\
                      intermediate.name))
        
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

    def add_comment(self, comment_str):
        """
        Add comment to ODE
        """
        check_arg(comment_str, str, context=ODE.add_comment)
        comment = Comment(comment_str, self._present_component.name)
        self._intermediates.append(comment)
        self._comments.append(comment)

    def add_subode(self, subode, prefix=None, components=None,
                   skip_duplicated_global_parameters=True):
        """
        Load an ODE and add it to the present ODE

        Argument
        --------
        subode : str, ODE
            The subode which should be added. If subode is a str an
            ODE stored in that file will be loaded. If it is an ODE it will be
            added directly to the present ODE.
        prefix : str (optional)
            A prefix which all state and parameters are prefixed with. If not
            given the name of the subode will be used as prefix. If set to
            empty string, no prefix will be used.
        components : list, tuple of str (optional)
            A list of components which will be extracted and added to the present
            ODE. If not given the whole ODE will be added.
        skip_duplicated_global_parameters : bool (optional)
            If true global parameters and variables will be skipped if they exists
            in the present model.
        """

        # If ode is given directly 
        if isinstance(subode, ODE):
            ode = subode
            
        else:
            # If not load external ODE
            from loadmodel import load_ode
            ode = load_ode(subode)
        
        components = components or []
        prefix = ode.name if prefix is None else prefix

        # Postfix prefix with "_" if prefix is not ""
        if prefix:
            prefix += "_"
        
        # If extracting only a certain components
        if components:
            ode = ode.extract_components(ode.name, *components)

        # Namespace for returning new symbols
        ns = {}

        # Collect prefixed states and parameters to be used to substitute
        # the intermediate and derivative expressions
        prefix_subs = {}

        # Add Markov models
        markov_model_actions = []
        for mm in ode.markov_models:
            
            # FIXME: Allow propagating of Parameter information
            mm_states = {}
            for state in mm.states:
                prefix_subs[state.sym] = ModelSymbol(prefix+state.name, self.name)
                param_repr = repr(state.param).replace("Slave", "Scalar")
                param_repr = param_repr.split(", name=")[0] + ")"
                mm_states[prefix+state.name] = eval(param_repr)
            
            obj = self.add_markov_model(prefix+mm.name, component=mm.component,
                                        algebraic_sum=mm._algebraic_sum, **mm_states)

            # Add rates (to be added later)
            for mm_states, expr in mm._rates.items():
                mm_states = tuple(prefix_subs[state] for state in mm_states)
                markov_model_actions.append((obj, mm_states, expr.expr))

            # Update namespace
            ns.update((state.name, state.sym) for state in obj.states)
        
        # Add prefixed states
        for state in ode.states:
            if state.slaved:
                continue
                
            prefix_subs[state.sym] = ModelSymbol(prefix+state.name, self.name)
            prefix_subs[state.derivative.sym] = ModelSymbol(\
                "d"+prefix+state.name+"_dt", self.name)
            
            if prefix+state.name in self._states:
                error("State with name {0} already exist in ODE".format(\
                    prefix+state.name))

            # Add the state
            sym = self.add_state(prefix+state.name, state.param, \
                                 state.derivative.param, state.component,
                                 replace_level="global_params")
            ns[prefix+state.name] = sym

        # Add prefixed parameters
        for parameter in ode.parameters:
            if parameter.slaved:
                continue
                
            prefix_subs[parameter.sym] = ModelSymbol(prefix+parameter.name, \
                                                     self.name)
        
            # If variable name already excist in this ODE do not add it
            # This will implicitly exchange all variables with corresponding
            # states and parameters from this ODE
            obj = self.get_object(parameter.name) or self._intermediates.get(\
                parameter.name)

            # If duplicated version already excists
            if obj and parameter.component == ode.name and \
                   skip_duplicated_global_parameters:
                debug("Skipping global {0} {1} as a {2} with same name already "\
                      "excists.".format(type(parameter).__name__, parameter.name, \
                                                    type(obj).__name__))
                continue

            # If component name is the same as the ode name, change
            # component name to this ODE name
            component = self.name if parameter.component == ode.name \
                        else parameter.component
            prefix_ = prefix if parameter.component == ode.name else ""
            
            # Add the parameter
            sym = self.add_parameter(prefix_+parameter.name, parameter.param, \
                                     component=component, \
                                     replace_level="global_params")
            ns[prefix+parameter.name] = sym

        # Add variables if not name already excisting in this ODE
        for variable in ode.variables:
            if variable.slaved:
                continue

            # If variable name already excist in this ODE do not add it
            # This will implicitly exchange all variables with corresponding
            # states and parameters from this ODE
            obj = self.get_object(variable.name) or self._intermediates.get(\
                variable.name)

            # If duplicated version already excists
            if obj and variable.component == ode.name and \
                   skip_duplicated_global_parameters:
                debug("Skipping global {0} {1} as a {2} with same name already "\
                      "excists.".format(type(variable).__name__, variable.name, \
                                            type(obj).__name__))
                continue

            # If component name is the same as the ode name, change
            # component name to this ODE name
            component = self.name if variable.component == ode.name \
                        else variable.component
            prefix_ = prefix if parameters.component == ode.name else ""
            
            # Add the un-prefixed variable
            sym = self.add_variable(prefix_+variable.name, variable.param,
                                    component=component,
                                    replace_level="global_params")
            ns[variable.name] = sym

        # Add intermediates
        for intermediate in ode.intermediates:
            if isinstance(intermediate, ODEComponent):
                self.set_component(intermediate.name)
            elif isinstance(intermediate, Comment):
                self.add_comment(intermediate.name)

            # Do not add slaved intermediates
            elif intermediate.slaved:
                    continue
            else:
                # Iterate over dependencies and check if we need to subs name
                # with prefixed name
                subs_list = []
                if prefix != "":
                    for obj_dep in intermediate.object_dependencies:
                        if obj_dep.sym in prefix_subs:
                            subs_list.append((obj_dep.sym, \
                                              prefix_subs[obj_dep.sym]))

                sym = self.add_intermediate(intermediate.name, \
                                            intermediate.expr.subs(subs_list), \
                                            replace_level="global_params")
                ns[intermediate.name] = sym
        
        # Add all rates for the markov models
        for mm, states, expr in markov_model_actions:
            if prefix != "":
                subs_list = []
                for sym in iter_symbol_params_from_expr(expr):
                    if sym in prefix_subs:
                        subs_list.append((sym, prefix_subs[sym]))
                expr = expr.subs(subs_list)

            mm[states] = expr

        # Add derivatives
        for derivative in ode._derivative_expressions:

            if derivative.slaved:
                continue
            
            # Iterate over dependencies and check if we need to subs name
            # with prefixed name
            subs_list = []
            der_subs_list = []
            if prefix != "":
                for obj_dep in derivative.object_dependencies:
                    if obj_dep.sym in prefix_subs:
                        subs_list.append((obj_dep.sym, \
                                          prefix_subs[obj_dep.sym]))
                for der in derivative.stripped_derivatives:
                    der_subs_list.apped((der, prefix_subs[der]))

            self.diff(derivative.derivatives.subs(der_subs_list), \
                      derivative.expr.subs(subs_list))
            
        # If set to return namespace
        if self._return_namespace:
            return ns

    def present_component(self, component=None):
        """
        Return the present component

        Arguments
        ---------
        component : str (optional)
            A component name. If not given the present component will be used.
        """
        if not component:
            return self._present_component

        # Check if component is already registered
        comp = self._components.get(component)
        if comp is None:
            comp = ODEComponent(component, self)
            self._components[component] = comp

        # Update present component
        self._present_component = comp

        return self._present_component
        
    def set_component(self, component):
        """
        Set present component
        """
        check_arg(component, str, 0, ODE.set_component)

        if component in self._components_set:
            error("Component can only be set once per ODE")

        # Get present component
        comp = self.present_component(component)

        # Register that this component has been set
        self._components_set.append(comp.name)

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

    def algebraic(self, state, expr, component=""):
        """
        Register an expression for a state given by an algebraic expression

        Arguments
        ---------
        state : sympy.Basic
            A linear expression of StateDerivative symbols
        expr : Sympy expression of ModelSymbols
            The derivative expression
        component : str (optional)
            The component will be determined automatically if the
            DerivativeExpression is an Algebraic expression
        """
        pass

    def diff(self, derivatives, expr, component=""):
        """
        Register an expression for state derivatives

        Arguments
        ---------
        derivatives : sympy.Basic
            A linear expression of StateDerivative symbols
        expr : Sympy expression of ModelSymbols
            The derivative expression
        component : str (optional)
            The component will be determined automatically if the
            DerivativeExpression is an Algebraic expression
        """

        timer = Timer("diff")
        derivative_expression = DerivativeExpression(\
            derivatives, expr, self, component)
        
        # Store expressions
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

    def save(self, basename):
        """
        Save ODE to file

        Arguments
        ---------
        basename : str
            The basename of the file which the ode will be saved to
        """
        from modelparameters.codegeneration import sympycode
        from gotran.codegeneration.codegenerator import CodeGenerator

        if not self.is_complete:
            error("ODE need to be complete to be saved to file.")

        lines = []

        # Add Markov models
        markov_model_actions = dict()

        # Write all States, Parameters and Variables
        for comp in self.components.values():

            # Markov models
            if comp.markov_models:
                for mm in comp.markov_models:
                    lines.append("markov_model(\"{0}\", \"{1}\",{2}".format(\
                        mm.name, mm.component, \
                        "" if mm._algebraic_sum is None else \
                        " algebraic_sum={0},".format(mm._algebraic_sum)))

                    # Add states
                    for state in sorted(mm.states, \
                                        lambda x,y:cmp(x.name, y.name)):
                        
                        # Get param repr and replace possible Slave with Scalar
                        param_repr = repr(state.param).replace("Slave", \
                                                               "Scalar")
                        param_repr = param_repr.split(", name=")[0] + ")"
                        lines.append("             {0}={1},".format(\
                            state.name, param_repr))

                    lines[-1] += ")"
                    lines.append("")

                    # Add the actions, to be added later
                    markov_model_actions[mm.component] = []
                    for states, expr in mm._rates.items():
                        markov_model_actions[mm.component].append((\
                            mm.name, states, expr.expr))

            # States
            if comp.states:

                lines.append("states(\"{0}\", ".format(comp.name))
                for state in comp.states.values():

                    if state.slaved:
                        continue

                    # Param repr and strip name and symname
                    param_repr = repr(state.param)
                    param_repr = param_repr.split(", name=")[0] + ")"
                
                    lines.append("       {0}={1},".format(state.name, param_repr))
                lines[-1] += ")"
                lines.append("")

            # Parameters
            if comp.parameters:
                lines.append("parameters(\"{0}\", ".format(comp.name))
                for param in comp.parameters.values():

                    if param.slaved:
                        continue

                    # Param repr and strip name and symname
                    param_repr = repr(param.param)
                    param_repr = param_repr.split(", name=")[0] + ")"
                
                    lines.append("       {0}={1},".format(param.name, param_repr))
                lines[-1] += ")"
                lines.append("")

            # Check for component containing time and dt
            # These should not be extracted
            if comp.name == self.name and len(comp.variables) == 2:
                continue

            # Variables
            if comp.variables:
                lines.append("variables(\"{0}\", ".format(comp.name))
                for variable in comp.variables.values():

                    # Do not include time or dt variables
                    if comp.name == self.name and variable.name in \
                           ["time", "dt"]:
                        continue
                    
                    if variable.slaved:
                        continue

                    # Param repr and strip name and symname
                    param_repr = repr(variable.param)
                    param_repr = param_repr.split(", name=")[0] + ")"
                
                    lines.append("       {0}={1},".format(variable.name, param_repr))
                lines[-1] += ")"
                lines.append("")

        # Write all Intermediates
        present_component = None
        for intermediate in self.intermediates:

            if isinstance(intermediate, ODEComponent):

                # Add Markov model rates at end of component
                if present_component in markov_model_actions:
                    lines.append("")
                    mm_actions = markov_model_actions.pop(present_component)
                    for mm, states, expr in mm_actions:
                        lines.append("{0}[{1}, {2}] = {3}".format(\
                            mm, states[0], states[1], sympycode(expr)))

                # Add next component
                lines.append("")
                lines.append("component(\"{0}\")".format(intermediate.name))
                present_component = intermediate.name
                
            elif isinstance(intermediate, Comment):
                lines.append("comment(\"{0}\")".format(intermediate.name))

            elif isinstance(intermediate, Intermediate):
                if not intermediate.slaved:
                    lines.append(sympycode(intermediate.expr, \
                                           intermediate.name))
                
        # Add any Markov models that is left
        for mm_actions in markov_model_actions.values():
            lines.append("")
            for mm, states, expr in mm_actions:
                lines.append("{0}[{1}, {2}] = {3}".format(\
                    mm, states[0], states[1], sympycode(expr)))

        lines.append("")

        # Write all Derivatives
        for der in self._derivative_expressions:

            # Do not write slaved derivatives
            if der.slaved:
                continue
            
            if der.num_derivatives == 1:
                lines.append(sympycode(der.expr, der.name))
            else:
                lines.append("diff({0}, {1})".format(der.name, der.expr))

        # Use Python code generator to indent outputted code
        # Write to file
        open(basename+".ode", "w").write("\n".join(\
            CodeGenerator.indent_and_split_lines(lines)))

    def extract_components(self, name, *components):
        """
        Create an ODE from a number of components

        Returns an ODE including the components

        Argument
        --------
        name : str
            The name of the created ODE
        components : str
            A variable len tuple of str describing the components
        """
        check_arg(name, str, 0)
        check_arg(components, tuple, 1, itemtypes=str)

        components = list(components)

        collected_components = ODEObjectList()
        
        # Collect components and check that the ODE has the components
        for original_component in self._components.values():

            if original_component.name in components:
                components.pop(components.index(original_component.name))
                collected_components.append(original_component)

        # Check that there are no components left
        if components:
            if len(components)>1:
                error("{0} are not a components of this ODE.".format(\
                    ", ".join("'{0}'".format(comp) for comp in components)))
            else:
                error("'{0}' is not a component of this ODE.".format(\
                    components[0]))
                
        # Collect intermediates
        intermediates = ODEObjectList()
        for intermediate in self.intermediates:
            
            # If Component 
            if isinstance(intermediate, ODEComponent):
                
                # Check if component is in components
                if intermediate.name in collected_components:
                    intermediates.append(intermediate)
            
            # Check of intermediate is in components
            elif intermediate.component in collected_components:
                intermediates.append(intermediate)

        # Collect states, parameters and derivatives
        states, parameters, derivatives, variables, markov_models = \
                ODEObjectList(), ODEObjectList(), ODEObjectList(), \
                ODEObjectList(), ODEObjectList()
        external_object_dep = set()
        for comp in collected_components:
            markov_models.extend(comp.markov_models)
            states.extend(comp.states.values())
            parameters.extend(comp.parameters.values())
            variables.extend(comp.variables.values())
            derivatives.extend(comp.derivatives)
            external_object_dep.update(comp.external_object_dep)

        # Check for dependencies
        for obj in external_object_dep:
            if (obj not in states) and (obj not in parameters) and \
                   (obj not in intermediates):

                # Create a Variable to replace an external object dependency
                # Put the Variable in the main ODE component
                variables.append(Variable(obj.name, obj.value, \
                                          name, name))

        # Create return ODE
        ode = ODE(name)

        # Add Markov models
        markov_model_actions = []
        for mm in markov_models:

            # FIXME: Allow propagating of Parameter information
            mm_states = dict((state.name, state.value) \
                             for state in mm.states)
            ode.add_markov_model(mm.name, component=mm.component, \
                                 algebraic_sum=mm._algebraic_sum, \
                                 **mm_states)

            obj = ode.get_object(mm.name)

            # Add rates
            for mm_states, expr in mm._rates.items():
                markov_model_actions.append((obj, mm_states, expr.expr))

        # Add states
        for state in states:

            # Do not add slaved states
            if state.slaved:
                continue
            
            ode.add_state(state.name, state.init, state.derivative.init, \
                          state.component)
        
        # Add parameters
        for param in parameters:

            # Do not add slaved parameters
            if param.slaved:
                continue
            
            ode.add_parameter(param.name, param.init, param.component)

        # Add variables
        for variable in variables:

            # Do not add slaved variables
            if variable.slaved:
                continue
            
            if variable.name in ["time", "dt"]:
                continue
            
            ode.add_variable(variable.name, variable.init, variable.component)

        # Add intermediates
        for intermediate in intermediates:

            if isinstance(intermediate, ODEComponent):
                ode.set_component(intermediate.name)
            elif isinstance(intermediate, Comment):
                ode.add_comment(intermediate.name)

            # Do not add slaved intermediates
            elif intermediate.slaved:
                    continue
            else:
                ode.add_intermediate(intermediate.name,\
                                     intermediate.expr)

        # Add all rates for the markov models
        for mm, states, expr in markov_model_actions:
            mm[states] = expr

        # Add derviatives
        for derivative in derivatives:

            if derivative.slaved:
                continue
            
            ode.diff(derivative.derivatives, derivative.expr)

        # Finalize the ode before returning it
        ode.finalize()
        
        # Return the ode
        return ode
        
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
        self._derivative_expressions = ODEObjectList()
        self._derivative_states = set() # FIXME: No need for a set here...
        self._algebraic_states = set()

        # Collection of intermediate stuff
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
        self.add_variable("time", 0.0, self._default_component.name)
        self.add_variable("dt", 0.1, self._default_component.name)
        
    def _add_entities(self, component, items, entity):
        """
        Help function for determine if each entity in the kwargs is unique
        and to check the type of the given default value
        """
        assert(entity in ["state", "parameter", "variable"])
    
        # Get caller frame
        # namespace = _get_load_namespace()
    
        # Get add method
        add = getattr(self, "add_{0}".format(entity))
        
        ns = {}

        # Symbol and value dicts
        for name, value in items:
    
            # Add the symbol
            sym = add(name, value, component=component)
            ns[name] = sym

        # If set to return namespace
        if self._return_namespace:
            return ns

    def _remove_duplicate(self, dup_obj, remove_level):
        """
        Remove an already registered object 
        """
        if not isinstance(dup_obj, (State, Parameter, Variable)):
            error("Trying to remove '{0}' which is a '{1}' is not allowed. "\
                  "Can only remove States, Parameters and Variables.".format(\
                      dup_obj, type(dup_obj).__name__))

        allowed_levels = ["none", "global_params", "any"]
        if remove_level not in allowed_levels:
            value_error("expected 'remove_level' argument to be one of: {0}".format(\
                ", ".join("'{0}'" for what in allowed_levels)))
            
        if remove_level == "none":
            return False
            
        # If the name of the registered object is the same as a Variable or
        # Parameter in the component with the same name as the ODE (global),
        # we overwrite that Variable/Parameter with the new object
        if remove_level == "global_params" and not \
               (self.is_global(dup_obj) and isinstance(\
            dup_obj, (Variable, Parameter))):
            return False
            
        debug("Removing {0}: '{1}' with a {2} with same name".format(\
            type(dup_obj).__name__, dup_obj.name, type(dup_obj).__name__))

        # Remove all traces of old object
        self._all_single_ode_objects.pop(dup_obj.name)
        if isinstance(dup_obj, Variable):
            self._variables.remove(dup_obj)
        elif isinstance(dup_obj, State):
            self._states.remove(dup_obj)
            if dup_obj.is_field:
                self._field_states.remove(dup_obj)
            self._all_single_ode_objects.pop(dup_obj.derivative)

            # Remove any derivative expressions that depends on the state
            for der_expr in self._derivative_expr[:]:
                if dup_obj in der_expr.states:
                    self._derivative_expr.pop(der_expr)

            # FIXME: Add logic for removing algebraic expression for state
            # FIXME: Add logic for removing slaved entities
            
        elif isinstance(dup_obj, Parameter):
            self._parameters.remove(dup_obj)
            if dup_obj.is_field:
                self._field_parameters.remove(dup_obj)

        # Clean up components and component dependencies
        self._components[dup_obj.component].remove(dup_obj)

        return True
        
    def _register_object(self, obj, replace_level="none"):
        """
        Register an ODEObject (only used for states, parameters, variables,
        and state_derivatives)

        Arguments
        ---------
        obj : ODEObject
            The object to be registered
        replace_level : str (optional)
            If not "none", replace can be either "global_params" or "any", meaning that
            either global or any object can be replaced by the added intermediate
        """
        
        timer = Timer("Register obj")
        assert(isinstance(obj, (State, Parameter, Variable, StateDerivative, \
                                MarkovModel)))

        check_kwarg(replace_level, "replace_level", str)

        # Check for existing object
        dup_obj = self._all_single_ode_objects.get(obj.name) or \
                  self._intermediates.get(obj.name)

        # If object already exists
        if dup_obj and not self._remove_duplicate(dup_obj, replace_level):
            error("Cannot register a {0}. A {1} with name '{2}' is "\
                  "already registered in this ODE.".format(\
                      type(obj).__name__, type(dup_obj).__name__, obj.name))
        
        # Get present component
        comp = self.present_component(obj.component)

        # Add object to component if not StateDerivative
        if not isinstance(obj, StateDerivative):
            comp.append(obj)

        # Register the object
        self._all_single_ode_objects[obj.name] = obj

        # Register the symbol of the Object or just the object as an attribute
        if isinstance(obj, ValueODEObject):
            self.__dict__[obj.name] = obj.sym
        else:
            self.__dict__[obj.name] = obj

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
        
        # Assume that we are registering an intermediate
        if isinstance(value, scalars) or (isinstance(value, sp.Basic) \
                                and any(isinstance(atom, ModelSymbol)\
                                        for atom in value.atoms())):
            self.add_intermediate(name, value)
        else:
            debug("Not registering: {0} as attribut. It does not contain "\
                  "any symbols or scalars.".format(name))

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
        for mine, obj in self._all_single_ode_objects.items():

            # Check that all mine obj excist in other
            if mine not in other._all_single_ode_objects:
                info("{0} {1} not in {1}".format(type(obj).__name__, \
                                                 mine, other.name))
                return False

            # Check same Type
            other_obj = other._all_single_ode_objects[mine]
            if not isinstance(other_obj, type(obj)):
                info("{0} in {1} is a {2} and in {3} it is a {4}".format(\
                    mine, self.name, type(obj).__name__), other.name, \
                     type(other_obj).__name__)
                return False

        # Check that all other obj excist in mine
        for other_name, obj in other._all_single_ode_objects.items():
            if other_name not in self._all_single_ode_objects:
                info("{0} {1} not in {2}".format(type(obj).__name__, \
                                                 other_name, self.name))
                return False

        for mine in self.intermediates:
            if mine not in other.intermediates:
                info("{0} not an intermediate in  {1}".format(\
                         mine.name, other.name))
                return False

        for other_inter in other.intermediates:
            if other_inter not in self.intermediates:
                info("{0} not an intermediate in  {1}".format(\
                         other_inter.name, self.name))
                return False

            if not isinstance(other_inter, Expression):
                continue

            # Compare the last intermediate
            mine_inter = self.intermediates.get(other_inter.name)
            other_inter = other.intermediates.get(other_inter.name)
            if abs(other_inter.param.value - mine_inter.param.value) > 1e-6:
                info("{0} : {1} != {2}".format(\
                    other_inter.name, other_inter.expr, mine_inter.expr))
                return False

        for mine in self._derivative_expressions:
            if mine not in other._derivative_expressions:
                info("{0} not a derivative in  {1}".format(\
                         mine.name, other.name))
                return False

        for other_der in other._derivative_expressions:
            if other_der not in self._derivative_expressions:
                info("{0} not a derivative in  {1}".format(\
                         other_der.name, self.name))
                return False

            mine_der = self._derivative_expressions.get(other_der.name)
            if abs(mine_der.param.value - other_der.param.value) > 1e-6:
                info("{0} : {1} != {2}".format(\
                         mine_der.name, mine_der.expr, other_der.expr))
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
    def comments(self):
        """
        Return a all comments
        """
        return self._comments

    @property
    def intermediates(self):
        """
        Return a all components
        """
        return self._intermediates

    @property
    def markov_models(self):
        """
        Return a all Markov models
        """
        return self._markov_models

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
    
    def has_component(self, component):
        """
        Return True if component is a registered ODEComponent
        """
        check_arg(component, str)

        return component in self._components
        
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

    def is_global(self, obj):
        """
        Return True if obj is part of the "global" component.

        The global component is the one with the same name as the ode.
        
        """
        if not isinstance(obj, ValueODEObject):
            return False
        return obj.component == self.name

    @property
    def is_complete(self):
        """
        Check that the ODE is complete
        """

        # Finalize before checking for completness
        self.finalize()
        
        states = self._states

        timer = Timer("Is complete")

        if not states:
            return False

        if len(states) > self.num_derivative_expr + self.num_algebraic_expr:
            # FIXME: Need a better name instead of xpressions...
            
            
            info("The ODE is under determined. The number of States are more "\
                 "than the number of derivative expressions.")
            
            return False

        if len(states) < self.num_derivative_expr + self.num_algebraic_expr:
            # FIXME: Need a better name instead of xpressions...
            info("The ODE is over determined. The number of States are less "\
                 "than the number of derivative expressions.")
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
                    if derivative[0].sym == state.sym:
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

    def finalize(self):
        """
        Finalize all finalizable objects
        """
        for mm in self._markov_models:
            mm.finalize()
        
