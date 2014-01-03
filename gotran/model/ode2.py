# Copyright (C) 2012-2014 Johan Hake
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
from collections import defaultdict
import weakref

from sympy.core.function import AppliedUndef

# ModelParameters imports
from modelparameters.sympytools import sp, symbols_from_expr
from modelparameters.codegeneration import sympycode, pythoncode
from modelparameters.utils import Timer

# Local imports
from gotran.common import error, debug, check_arg, check_kwarg, scalars
from gotran.model.odeobjects2 import *
from gotran.model.expressions2 import *
from gotran.model.odecomponent import *
from gotran.model.utils import PresentObjTuple

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


    def __new__(cls, *args, **kwargs):
        self = object.__new__(cls, *args, **kwargs)
        return self
    
    def __init__(self, name, ns=None):

        # Call super class with itself as parent component
        super(ODE, self).__init__(name, self)

        # Reset constructed attribute
        self._constructed = False

        # If namespace provided just keep a weak ref
        if ns is None:
            self._ns = {}
        else:
            self._ns = weakref.ref(ns) 

        # Add Time object
        # FIXME: Add information about time unit dimensions and make
        # FIXME: it possible to have different time names
        time = Time("t", "ms")
        self._time = time
        self.ode_objects.append(time)

        # Namespace, which can be used to eval an expression
        self.ns.update({"t":time.sym, "time":time.sym})
        
        # An list with all component names with expression added to them
        # The components are always sorted wrt last expression added
        self.all_expr_components_ordered = []

        # A dict with all components objects
        self.all_components = weakref.WeakValueDictionary()
        self.all_components[name] = self

        # An attribute keeping track of the present ODE component
        self._present_component = self.name

        # A dict with the present ode objects
        # NOTE: hashed by name so duplicated expressions are not stored
        self.present_ode_objects = {}
        self.present_ode_objects["t"] = PresentObjTuple(self._time, self)
        self.present_ode_objects["time"] = PresentObjTuple(self._time, self)
        #self.present_ode_objects = dict(t=(self._time, self), time=(self._time, self))

        # Keep track of duplicated expressions
        self.duplicated_expressions = defaultdict(list)

        # Keep track of expression dependencies and in what expression
        # an object has been used in
        self.expression_dependencies = defaultdict(set)
        self.object_used_in = defaultdict(set)

        # All expanded expressions
        self.expanded_expressions = dict()

        # Attributes which will be populated later
        self._body_expressions = None
        self._mass_matrix = None
        
        # Global finalized flag
        self._is_finalized_ode = False
        
        # Flag that the ODE is constructed
        self._constructed = True

    @property
    def ns(self):
        if isinstance(self._ns, dict):
            return self._ns

        # Check if ref in weakref is alive
        ns = self._ns()
        if isinstance(ns, dict):
            return ns

        # If not just return an empty dict
        return {}

    @property
    def present_component(self):
        """
        Return the present component
        """
        return self.all_components[self._present_component]

    def add_sub_ode(self, subode, prefix=None, components=None,
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
        pass

    def save(self, basename=None):
        """
        Save ODE to file

        Arguments
        ---------
        basename : str (optional)
            The basename of the file which the ode will be saved to, if not
            given the basename will be the same as the name of the ode.
        """

        if not self._is_finalized_ode:
            error("ODE need to be finalized to be saved to file.")
            
        lines = ["# Saved Gotran model"]

        comp_names = dict()
        
        basename = basename or self.name
        
        for comp in self.components:
            if comp == self:
                comp_name = ""
            else:
                present_comp = comp
                comps = [present_comp.name]
                while present_comp.parent != self:
                    present_comp = present_comp.parent
                    comps.append(present_comp.name)
                comp_name = ", ".join("\"{0}\"".format(name) for name in reversed(comps))

            comp_names[comp] = comp_name
                    
            states = ["{0}={1},".format(obj.name, obj.param.repr(\
                include_name=False)) for obj in comp.ode_objects \
                      if isinstance(obj, State)]
            parameters = ["{0}={1},".format(obj.name, obj.param.repr(\
                include_name=False)) for obj in comp.ode_objects \
                          if isinstance(obj, Parameter)]
            if states:
                lines.append("")
                if comp_name:
                    lines.append("states({0},".format(comp_name))
                else:
                    lines.append("states({0}".format(states.pop(0)))
                for state_code in states:
                    lines.append("       " + state_code)
                lines[-1] = lines[-1][:-1] + ")"

            if parameters:
                lines.append("")
                if comp_name:
                    lines.append("parameters({0},".format(comp_name))
                else:
                    lines.append("parameters({0}".format(parameters.pop(0)))
                for param_code in parameters:
                    lines.append("           " + param_code)
                lines[-1] = lines[-1][:-1] + ")"

        # Iterate over all components
        for comp_name in self.all_expr_components_ordered:

            comp = self.all_components[comp_name]

            comp_comment = "Intermediate expressions for the {0} "\
                           "component".format(comp.name)

            # Iterate over all objects of the component and save only expressions
            # and comments
            for obj in comp.ode_objects:

                # If saving an expression
                if isinstance(obj, Expression):

                    # If the component is a Markov model
                    if comp.rates:

                        # Do not save State derivatives
                        if isinstance(obj, StateDerivative):
                            continue

                        # Save rate expressions slightly different
                        elif isinstance(obj, RateExpression):
                            lines.append("rates[{0}, {1}] = {2}".format(\
                                sympycode(obj.states[0]), \
                                sympycode(obj.states[1]), \
                                sympycode(obj.expr)))
                            continue

                    # All other Expressions
                    lines.append("{0} = {1}".format(obj.name, \
                                                    sympycode(obj.expr)))

                # If saving a comment
                elif isinstance(obj, Comment):

                    # If comment is component comment
                    if str(obj) == comp_comment:
                        lines.append("")
                        comp_name = comp_names[comp] if comp_names[comp] \
                                    else "\"{0}\"".format(basename)
                        lines.append("component({0})".format(comp_name))

                    # Just add the comment
                    else:
                        lines.append("")
                        lines.append("comment(\"{0}\")".format(obj))

        lines.append("")
        # Use Python code generator to indent outputted code
        # Write to file
        from gotran.codegeneration.codegenerator2 import PythonCodeGenerator
        open(basename+".ode", "w").write("\n".join(\
            PythonCodeGenerator.indent_and_split_lines(lines)))
                    
    def register_ode_object(self, obj, comp):
        """
        Register an ODE object in the root ODEComponent
        """

        if self._is_finalized_ode and isinstance(obj, StateExpression):
            error("Cannot register a StateExpression, the ODE is finalized")

        # Check for existing object in the ODE
        duplication = self.present_ode_objects.get(obj.name)

        # If object with same name is already registered in the ode we
        # need to figure out what to do
        if duplication:

            dup_obj, dup_comp = duplication.obj, duplication.comp

            # If State, Parameter or DerivativeExpression we always raise an error
            if isinstance(dup_obj, State) and isinstance(obj, StateSolution):
                debug("Reduce state '{0}' to {1}".format(dup_obj, obj.expr))

            elif any(isinstance(oo, (State, Parameter, Time, DerivativeExpression,
                                     AlgebraicExpression, StateSolution)) \
                     for oo in [dup_obj, obj]):
                error("Cannot register {0}. A {1} with name '{2}' is "\
                      "already registered in this ODE.".format(\
                          type(obj).__name__, type(\
                              dup_obj).__name__, dup_obj.name))
            else:

                # Sanity check that both obj and dup_obj are Expressions
                assert all(isinstance(oo, (Expression)) for oo in [dup_obj, obj])

                # Get list of duplicated objects or an empy list
                dup_objects = self.duplicated_expressions[obj.name]
                if len(dup_objects) == 0:
                    dup_objects.append(dup_obj)
                dup_objects.append(obj)

        # Update global information about ode object
        self.present_ode_objects[obj.name] = PresentObjTuple(obj, comp)
        self.ns.update({obj.name : obj.sym})

        # If Expression
        if isinstance(obj, Expression):

            # Append the name to the list of all ordered components with
            # expressions. If the ODE is finalized we do not update components
            if not self._is_finalized_ode:
                self._handle_expr_component(comp, obj)

            # Add dependencies between registered comments and expressions so
            # they are carried over in Code components
            for comment in comp._local_comments:
                self.object_used_in[comment].add(obj)
                self.expression_dependencies[obj].add(comment)

            # Expand and add any derivatives in the expressions
            expression_added = False
            for der_expr in obj.expr.atoms(sp.Derivative):
                expression_added |= self._expand_single_derivative(comp, obj, der_expr)

            # If any expression was added we need to bump the count of the ODEObject
            if expression_added:
                obj._recount()

            # Expand the Expression
            self.expanded_expressions[obj.name] = self._expand_expression(obj)

            # If the expression is a StateSolution the state cannot have
            # been used previously
            if isinstance(obj, StateSolution) and \
                   self.object_used_in.get(obj.state):
                used_in = self.object_used_in.get(obj.state)
                error("A state solution cannot have been used in "\
                      "any previous expressions. {0} is used in: {1}".format(\
                          obj.state, used_in))

    def _handle_expr_component(self, comp, expr):
        """
        A help function to sort and add components in the ordered
        the intermediate expressions are added to the ODE
        """
        
        if len(self.all_expr_components_ordered) == 0:
            self.all_expr_components_ordered.append(comp.name)

            # Add a comment to the component
            comp.add_comment("Intermediate expressions for the {0} "\
                             "component".format(comp.name))

            # Recount the last added expression so the comment comes
            # infront of the expression
            expr._recount()

        # We are shifting expression components
        elif self.all_expr_components_ordered[-1] != comp.name:

            # Finalize the last component we visited
            self.all_components[\
                self.all_expr_components_ordered[-1]].finalize_component()
                
            # Append this component
            self.all_expr_components_ordered.append(comp.name)

            # Add a comment to the component
            comp.add_comment("Intermediate expressions for the {0} "\
                             "component".format(comp.name))

            # Recount the last added expression so the comment comes
            # infront of the expression
            expr._recount()
        
    def _expand_single_derivative(self, comp, obj, der_expr):
        """
        Expand a single derivative and register it as new derivative expression
        
        Returns True if an expression was actually added
        """

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
            return False

        # Get the expr and dependent variable objects
        expr_obj = self.present_ode_objects[sympycode(der_expr.args[0])].obj
        var_obj = self.present_ode_objects[sympycode(der_expr.args[1])].obj

        # If the dependent variable is time and the expression is a state
        # variable we raise an error as the user should already have created
        # the expression.
        if isinstance(expr_obj, State) and var_obj == self._time:
            error("The expression {0} is dependent on the state "\
                  "derivative of {1} which is not registered in this ODE."\
                  .format(obj, expr_obj))

        if not isinstance(expr_obj, Expression):
            error("Can only differentiate expressions or states. Got {0} as "\
                  "the derivative expression.".format(expr_obj))

        # If we get a Derivative(expr, t) we issue an error
        if isinstance(expr_obj, Expression) and var_obj == self._time:
            error("All derivative expressions of registered expressions "\
                  "need to be expanded with respect to time. Use "\
                  "expr.diff(t) instead of Derivative(expr, t) ")

        # Store expression
        comp.add_derivative(expr_obj, var_obj, expr_obj.expr.diff(var_obj.sym))

        return True

    def _expand_expression(self, obj):

        timer = Timer("Expanding expression")

        # FIXME: We need to wait for the expanssion of all expressions...
        assert isinstance(obj, Expression)

        # Iterate over dependencies in the expression
        expression_subs_dict = {}
        for sym in symbols_from_expr(obj.expr, include_derivatives=True):

            dep_obj = self.present_ode_objects[sympycode(sym)]

            if dep_obj is None:
                error("The symbol '{0}' is not declared within the '{1}' "\
                      "ODE.".format(sym, self.name))

            # Expand dep_obj
            dep_obj, dep_comp = dep_obj.obj, dep_obj.comp

            # Store object dependencies
            self.expression_dependencies[obj].add(dep_obj)
            self.object_used_in[dep_obj].add(obj)

            # Expand dependent expressions
            if isinstance(dep_obj, Expression):

                # Collect intermediates to be used in substitutions below
                expression_subs_dict[dep_obj.sym] = self.expanded_expressions[dep_obj.name]

        return obj.expr.xreplace(expression_subs_dict)

    @property
    def mass_matrix(self):
        """
        Return the mass matrix as a sympy.Matrix
        """

        if not self.is_finalized:
            error("The ODE must be finalized")
            
        if not self._mass_matrix:
        
            state_exprs = self.state_expressions
            N = len(state_exprs)
            self._mass_matrix = sp.Matrix(N, N, lambda i, j : 1 if i==j and \
                        isinstance(state_exprs[i], StateDerivative) else 0)
            
        return self._mass_matrix

    @property
    def is_dae(self):
        """
        Return True if ODE is a DAE
        """
        if not self.is_complete:
            error("The ODE is not complete")

        return any(isinstance(expr, AlgebraicExpression) for expr in \
                   self.state_expressions)

    def finalize(self):
        """
        Finalize the ODE
        """
        for comp in self.components:
            comp.finalize_component()
            
        self._is_finalized_ode = True
        self._present_component = self.name

    def signature(self):
        """
        Return a signature uniquely defining the ODE
        """
        import hashlib

        def_list = []
        for comp in self.components:
            if self != comp:
                def_list.append(str(comp))
            
            # Sort wrt stringified states and parameters avoiding trouble with
            # random ordering of **kwargs
            def_list += sorted([repr(state.param) for state in comp.full_states])
            def_list += sorted([repr(param.param) for param in comp.parameters])
            def_list += [str(expr.expr) for expr in comp.intermediates]
        
            # Sort state expressions wrt stringified state names
            def_list += [str(expr.expr) for expr in sorted(\
                self.state_expressions, cmp=lambda o0, o1: cmp(\
                    str(o0.state), str(o1.state)))]

        h = hashlib.sha1()
        h.update(";".join(def_list))
        return h.hexdigest()
