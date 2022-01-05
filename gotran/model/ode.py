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

import weakref
from collections import defaultdict
from functools import cmp_to_key

# System imports
from pathlib import Path

from modelparameters.codegeneration import sympycode

# Local imports
from modelparameters.logger import debug, error

# ModelParameters imports
from modelparameters.sympytools import sp
from modelparameters.utils import Timer, check_arg
from modelparameters.sympy.core.function import AppliedUndef

from .expressions import (
    AlgebraicExpression,
    DerivativeExpression,
    Derivatives,
    Expression,
    Intermediate,
    RateExpression,
    State,
    StateDerivative,
    StateExpression,
    StateSolution,
    recreate_expression,
)
from .odecomponent import Comment, ODEComponent
from .odeobjects import Dt, Parameter, Time, cmp


class ODE(ODEComponent):
    """
    Root ODEComponent

    Arguments
    ---------
    name : str
        The name of the ODE
    ns : dict (optional)
        A namespace which will be filled with declared ODE symbols
    """

    def __new__(cls, *args, **kwargs):
        self = object.__new__(cls)

        return self

    def __init__(self, name, ns=None):

        # Call super class with itself as parent component
        super(ODE, self).__init__(name, self)

        # Turn off magic attributes (see __setattr__ method) during
        # construction
        self._allow_magic_attributes = False

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

        dt = Dt("dt", "ms")
        self._dt = dt
        self.ode_objects.append(dt)

        # Add to object to component map
        self.object_component = weakref.WeakValueDictionary()
        self.object_component[self._time] = self
        self.object_component[self._dt] = self

        # Namespace, which can be used to eval an expression
        self.ns.update({"t": time.sym, "time": time.sym, "dt": dt.sym})

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
        self.present_ode_objects = dict(t=self._time, time=self._time, dt=self._dt)

        # Keep track of duplicated expressions
        self.duplicated_expressions = defaultdict(list)

        # Keep track of expression dependencies and in what expression
        # an object has been used in
        self.expression_dependencies = defaultdict(set)
        self.object_used_in = defaultdict(set)

        # All expanded expressions
        self._expanded_expressions = dict()

        # Attributes which will be populated later
        self._mass_matrix = None

        # Global finalized flag
        self._is_finalized_ode = False

        # Turn on magic attributes (see __setattr__ method)
        self._allow_magic_attributes = True

    @property
    def parameter_symbols(self):
        return [s.name for s in self.parameters]

    @property
    def component_names(self):
        return [s.name for s in self.components]

    def parameter_values(self):
        return [s.value for s in self.parameters]

    @property
    def state_symbols(self):
        return [s.name for s in self.states]

    def state_values(self):
        return [s.value for s in self.states]

    @property
    def intermediate_symbols(self):
        return [i.name for i in self.intermediates]

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

    def import_ode(self, ode, prefix="", components=None, **arguments):
        """
        Import a Model into the present Model

        Arguments
        ---------
        ode : str, gotran.ODE
            The ode which should be added. If ode is a str an
            ODE stored in that file will be loaded. If it is an ODE it will be
            added directly to the present ODE.
        prefix : str (optional)
            A prefix which all state, parameters and intermediates are
            prefixed with.
        components : list, tuple of str (optional)
            A list of components which will either be extracted or excluded
            from the imported model. If not given the whole ODE will be imported.
        arguments : dict (optional)
            Optional arguments which can control loading of model
        """

        timer = Timer("Import ode")  # noqa: F841

        components = components or []
        check_arg(ode, (str, Path, ODE), 0, context=ODE.import_ode)
        check_arg(prefix, str, 1, context=ODE.import_ode)
        check_arg(components, list, 2, context=ODE.import_ode, itemtypes=str)

        # If ode is given directly
        if isinstance(ode, (str, Path)):
            # If not load external ODE
            from gotran.model.loadmodel import load_ode

            ode = load_ode(ode, **arguments)

        # Postfix prefix with "_" if prefix is not ""
        if prefix and prefix[-1] != "_":
            prefix += "_"

        # If extracting only a certain components
        if components:
            ode = ode.extract_components(ode.name, *components)

        # Subs for expressions
        subs = {}
        old_new_map = {ode: self}

        def add_comp_and_children(added, comp):
            "Help function to recursively add components to ode"

            # Add states and parameters
            for obj in comp.ode_objects:

                # Check if obj already excists as a ODE parameter
                old_obj = self.present_ode_objects.get(str(obj))

                if old_obj and self.object_component[old_obj] == self:
                    new_name = obj.name
                else:
                    new_name = prefix + obj.name

                if isinstance(obj, State):
                    subs[obj.sym] = added.add_state(new_name, obj.param)

                elif isinstance(obj, Parameter):

                    # If adding an ODE parameter
                    if comp == ode:

                        # And parameter name already excists in the present ODE
                        if str(obj) in self.present_ode_objects:

                            # Skip it and add the registered symbol for
                            # substitution
                            subs[obj.sym] = self.present_ode_objects[str(obj)].sym

                        else:

                            # If not already present just add unprefixed name
                            subs[obj.sym] = added.add_parameter(obj.name, obj.param)

                    else:

                        subs[obj.sym] = added.add_parameter(new_name, obj.param)

            # Add child components
            for child in list(comp.children.values()):
                added_child = added.add_component(child.name)

                # Get corresponding component in present ODE
                old_new_map[child] = added_child

                add_comp_and_children(added_child, child)

        # Recursively add states and parameters
        add_comp_and_children(self, ode)

        # Iterate over all components to add expressions
        for comp_name in ode.all_expr_components_ordered:

            comp = ode.all_components[comp_name]

            comp_comment = f"Expressions for the {comp.name} component"

            # Get corresponding component in new ODE
            added = old_new_map[comp]

            # Iterate over all objects of the component and save only expressions
            # and comments
            for obj in comp.ode_objects:

                # If saving an expression
                if isinstance(obj, Expression):

                    # The new sympy expression
                    new_expr = obj.expr.xreplace(subs)

                    # If the component is a Markov model
                    if comp.rates:

                        # Do not save State derivatives
                        if isinstance(obj, StateDerivative):
                            continue

                        # Save rate expressions slightly different
                        elif isinstance(obj, RateExpression):

                            states = obj.states
                            added._add_single_rate(
                                subs[states[0].sym],
                                subs[states[1].sym],
                                new_expr,
                            )
                            continue

                    # If no prefix we just add the expression by using setattr
                    # magic
                    if prefix == "":
                        setattr(added, str(obj), new_expr)
                        subs[obj.sym] = added.ode_objects.get(obj.name).sym

                    elif isinstance(obj, (StateExpression, StateSolution)):

                        state = subs[obj.state.sym]

                        if isinstance(obj, AlgebraicExpression):
                            subs[obj.sym] = added.add_algebraic(state, new_expr)

                        elif isinstance(obj, StateDerivative):
                            subs[obj.sym] = added.add_derivative(
                                state,
                                added.t,
                                new_expr,
                            )

                        elif isinstance(obj, StateSolution):
                            subs[obj.sym] = added.add_state_solution(state, new_expr)
                            print(repr(obj), obj.sym)
                        else:
                            error("Should not reach here...")

                    # Derivatives are tricky. Here the der expr and dep var
                    # need to be registered in the ODE already. But they can
                    # be registered with and without prefix, so we need to
                    # check that.
                    elif isinstance(obj, Derivatives):

                        # Get der_expr
                        der_expr = self.root.present_ode_objects.get(obj.der_expr.name)

                        if der_expr is None:

                            # If not found try prefixed version
                            der_expr = self.root.present_ode_objects.get(
                                prefix + obj.der_expr.name,
                            )

                            if der_expr is None:
                                if prefix:
                                    error(
                                        "Could not find expression: "
                                        "({0}){1} while adding "
                                        "derivative".format(prefix, obj.der_expr),
                                    )
                                else:
                                    error(
                                        "Could not find expression: "
                                        "{0} while adding derivative".format(
                                            obj.der_expr,
                                        ),
                                    )

                        dep_var = self.root.present_ode_objects.get(obj.dep_var.name)

                        if isinstance(dep_var, Time):
                            dep_var = self._time

                        elif dep_var is None:

                            # If not found try prefixed version
                            dep_var = self.root.present_ode_objects.get(
                                prefix + obj.dep_var.name,
                            )

                            if dep_var is None:
                                if prefix:
                                    error(
                                        "Could not find expression: "
                                        "({0}){1} while adding "
                                        "derivative".format(prefix, obj.dep_var),
                                    )
                                else:
                                    error(
                                        "Could not find expression: "
                                        "{0} while adding derivative".format(
                                            obj.dep_var,
                                        ),
                                    )

                        subs[obj.sym] = added.add_derivative(
                            der_expr,
                            dep_var,
                            new_expr,
                        )

                    elif isinstance(obj, Intermediate):

                        subs[obj.sym] = added.add_intermediate(
                            prefix + obj.name,
                            new_expr,
                        )

                    else:
                        error("Should not reach here!")

                # If saving a comment
                elif isinstance(obj, Comment) and str(obj) != comp_comment:
                    added.add_comment(str(obj))

    def save(self, basename=None):
        """
        Save ODE to file

        Arguments
        ---------
        basename : str (optional)
            The basename of the file which the ode will be saved to, if not
            given the basename will be the same as the name of the ode.
        """

        timer = Timer("Save " + self.name)  # noqa: F841

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
                comp_name = ", ".join(f'"{name}"' for name in reversed(comps))

            comp_names[comp] = comp_name

            states = [
                f"{obj.name}={obj.param.repr(include_name=False)},"
                for obj in comp.ode_objects
                if isinstance(obj, State)
            ]
            parameters = [
                f"{obj.name}={obj.param.repr(include_name=False)},"
                for obj in comp.ode_objects
                if isinstance(obj, Parameter)
            ]
            if states:
                lines.append("")
                if comp_name:
                    lines.append(f"states({comp_name},")
                else:
                    lines.append(f"states({states.pop(0)}")
                for state_code in states:
                    lines.append("       " + state_code)
                lines[-1] = lines[-1][:-1] + ")"

            if parameters:
                lines.append("")
                if comp_name:
                    lines.append(f"parameters({comp_name},")
                else:
                    lines.append(f"parameters({parameters.pop(0)}")
                for param_code in parameters:
                    lines.append("           " + param_code)
                lines[-1] = lines[-1][:-1] + ")"

        # Iterate over all components
        for comp_name in self.all_expr_components_ordered:

            comp = self.all_components[comp_name]

            comp_comment = f"Expressions for the {comp.name} component"

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
                            lines.append(
                                "rates[{0}, {1}] = {2}".format(
                                    sympycode(obj.states[0]),
                                    sympycode(obj.states[1]),
                                    sympycode(obj.expr),
                                ),
                            )
                            continue

                    # All other Expressions
                    lines.append(f"{obj.name} = {sympycode(obj.expr)}")

                # If saving a comment
                elif isinstance(obj, Comment):

                    # If comment is component comment
                    if str(obj) == comp_comment:
                        lines.append("")
                        comp_name = (
                            comp_names[comp] if comp_names[comp] else f'"{basename}"'
                        )
                        lines.append(f"expressions({comp_name})")

                    # Just add the comment
                    else:
                        lines.append("")
                        lines.append(f'comment("{obj}")')

        lines.append("")
        # Use Python code generator to indent outputted code
        # Write to file
        from gotran.codegeneration.codegenerators import PythonCodeGenerator

        with open(basename + ".ode", "w") as f:
            f.write("\n".join(PythonCodeGenerator.indent_and_split_lines(lines)))

    def register_ode_object(self, obj, comp, dependent=None):
        """
        Register an ODE object in the root ODEComponent
        """

        from modelparameters.sympytools import symbols_from_expr

        if self._is_finalized_ode and isinstance(obj, StateExpression):
            error("Cannot register a StateExpression, the ODE is finalized")

        # Check for existing object in the ODE
        dup_obj = self.present_ode_objects.get(obj.name)

        # If object with same name is already registered in the ode we
        # need to figure out what to do
        if dup_obj:

            try:
                dup_comp = self.object_component[dup_obj]
            except KeyError:
                dup_comp = None

            # If a state is substituted by a state solution
            if isinstance(dup_obj, State) and isinstance(obj, StateSolution):
                debug(f"Reduce state '{dup_obj}' to {obj.expr}")

            # If duplicated object is an ODE Parameter and the added object is
            # either a State or a Parameter we replace the Parameter.
            elif (
                isinstance(dup_obj, Parameter)
                and dup_comp == self
                and comp != self
                and isinstance(obj, (State, Parameter))
            ):

                timer = Timer("Replace objects")  # noqa: F841

                # Remove the object
                self.ode_objects.remove(dup_obj)

                # FIXME: Do we need to recreate all expression the objects is used in?
                # Replace the object from the object_used_in dict and update
                # the correponding expressions
                subs = {dup_obj.sym: obj.sym}
                subs = {}

                # Recursively replace object dependencies
                self._replace_object(dup_obj, obj, subs)

                # for expr in self.object_used_in[dup_obj]:
                #    updated_expr = recreate_expression(expr, subs)
                #    self.object_used_in[obj].add(updated_expr)
                #
                #    # Exchange and update the dependencies
                #    self.expression_dependencies[expr].remove(dup_obj)
                #    self.expression_dependencies[expr].add(obj)
                #
                #    # FIXME: Do not remove the dependencies
                #    #self.expression_dependencies[updated_expr] = \
                #    #            self.expression_dependencies.pop(expr)
                #    self.expression_dependencies[updated_expr] = \
                #                self.expression_dependencies[expr]
                #
                #    # Find the index of old expression and exchange it with updated
                #    old_comp = self.object_component[expr]
                #    ind = old_comp.ode_objects.index(expr)
                #    old_comp.ode_objects[ind] = updated_expr
                #
                ## Remove information about the replaced objects
                # self.object_used_in.pop(dup_obj)

            # If duplicated object is an ODE Parameter and the added
            # object is an Intermediate we raise an error.
            elif (
                isinstance(dup_obj, Parameter)
                and dup_comp == self
                and isinstance(obj, Expression)
            ):
                error(
                    "Cannot replace an ODE parameter with an Expression, "
                    "only with Parameters and States.",
                )

            # If State, Parameter or DerivativeExpression we always raise an error
            elif any(
                isinstance(
                    oo,
                    (
                        State,
                        Parameter,
                        Time,
                        Dt,
                        DerivativeExpression,
                        AlgebraicExpression,
                        StateSolution,
                    ),
                )
                for oo in [dup_obj, obj]
            ):
                error(
                    "Cannot register {0}. A {1} with name '{2}' is "
                    "already registered in this ODE.".format(
                        type(obj).__name__,
                        type(dup_obj).__name__,
                        dup_obj.name,
                    ),
                )
            else:

                # Sanity check that both obj and dup_obj are Expressions
                assert all(isinstance(oo, (Expression)) for oo in [dup_obj, obj])

                # Get list of duplicated objects or an empy list
                dup_objects = self.duplicated_expressions[obj.name]
                if len(dup_objects) == 0:
                    dup_objects.append(dup_obj)
                dup_objects.append(obj)

        # Update global information about ode object
        self.present_ode_objects[obj.name] = obj
        self.object_component[obj] = comp
        self.ns.update({obj.name: obj.sym})

        # If Expression
        if isinstance(obj, Expression):

            # Append the name to the list of all ordered components with
            # expressions. If the ODE is finalized we do not update components
            if not self._is_finalized_ode:
                self._handle_expr_component(comp, obj)

            # Expand and add any derivatives in the expressions
            expression_added = False
            replace_dict = {}
            derivative_expression_list = list(obj.expr.atoms(sp.Derivative))
            derivative_expression_list.sort(key=lambda e: e.sort_key())
            for der_expr in derivative_expression_list:
                expression_added |= self._expand_single_derivative(
                    comp,
                    obj,
                    der_expr,
                    replace_dict,
                    dependent,
                )

            # If expressions need to be re-created
            if replace_dict:
                obj.replace_expr(replace_dict)

            # If any expression was added we need to bump the count of the ODEObject
            if expression_added:
                obj._recount(dependent=dependent)

            # Add dependencies between the last registered comment and
            # expressions so they are carried over in Code components
            if comp._local_comments:
                self.object_used_in[comp._local_comments[-1]].add(obj)
                self.expression_dependencies[obj].add(comp._local_comments[-1])

            # Get expression dependencies
            for sym in symbols_from_expr(obj.expr, include_derivatives=True):

                dep_obj = self.present_ode_objects[sympycode(sym)]

                if dep_obj is None:
                    error(
                        "The symbol '{0}' is not declared within the '{1}' "
                        "ODE.".format(sym, self.name),
                    )

                # Store object dependencies
                self.expression_dependencies[obj].add(dep_obj)
                self.object_used_in[dep_obj].add(obj)

            # If the expression is a StateSolution the state cannot have
            # been used previously
            if isinstance(obj, StateSolution) and self.object_used_in.get(obj.state):
                used_in = self.object_used_in.get(obj.state)
                error(
                    "A state solution cannot have been used in "
                    "any previous expressions. {0} is used in: {1}".format(
                        obj.state,
                        used_in,
                    ),
                )

    def expanded_expression(self, expr):
        """
        Return the expanded expression.

        The returned expanded expression consists of the original
        expression given by it basics objects (States, Parameters and
        IndexedObjects)
        """

        timer = Timer("Expand expression")  # noqa: F841

        check_arg(expr, Expression)

        # First check cache
        exp_expr = self._expanded_expressions.get(expr)

        if exp_expr is not None:
            return exp_expr

        # If not recursively expand the expression
        der_subs = {}
        subs = {}
        for obj in self.expression_dependencies[expr]:

            if isinstance(obj, Derivatives):
                der_subs[obj.sym] = self.expanded_expression(obj)

            elif isinstance(obj, Expression):
                subs[obj.sym] = self.expanded_expression(obj)

        # Do the substitution
        exp_expr = expr.expr.xreplace(der_subs).xreplace(subs)
        self._expanded_expressions[expr] = exp_expr

        return exp_expr

    def extract_components(self, name, *components):
        """
        Create an ODE from a number of components

        Returns an ODE including the components

        Arguments
        ---------
        name : str
            The name of the created ODE
        components : str
            A variable len tuple of str describing the components
        """
        check_arg(name, str, 0)
        check_arg(components, tuple, 1, itemtypes=str)

        components = list(components)

        collected_components = set()

        if self.name in components:
            error("Can only extract sub component of this ODE.")

        # Collect components and check that the ODE has the components
        for original_component in self.components:

            # Collect components together with its children
            if original_component.name in components:

                components.remove(original_component.name)
                collected_components.update(original_component.components)

        # Check that there are no components left
        if components:
            if len(components) > 1:
                error(
                    "{0} are not a components of this ODE.".format(
                        ", ".join("'{0}'".format(comp) for comp in components),
                    ),
                )
            else:
                error(f"'{components[0]}' is not a component of this ODE.")

        # Collect dependencies
        included_objects = []
        dependencies = set()
        for comp in self.components:
            if comp in collected_components:
                included_objects.extend(comp.ode_objects)
                for expr in comp.ode_objects:
                    if isinstance(expr, Expression):
                        dependencies.update(self.expression_dependencies[expr])

        # Remove included objects from dependencies
        dependencies.difference_update(included_objects)

        # Create return ODE
        ode = ODE(name)

        # Add dependencies as parameters to return ODE
        subs = dict()
        for dep in dependencies:

            # Skip time
            if str(dep) in ["t", "time", "dt"]:
                continue
            subs[dep.sym] = ode.add_parameter(dep.name, dep.param.value)

        # Add components together with states and parameters to the ODE
        components = sorted(collected_components, reverse=True)
        old_new_map = dict()

        def add_comp_and_children(comp, components, parent):
            "Help function to add recursively components to ode"

            # Add component
            added = parent.add_component(comp.name)
            old_new_map[comp] = added

            # Add states and parameters
            for obj in comp.ode_objects:
                if isinstance(obj, State):
                    added.add_state(obj.name, obj.param)
                elif isinstance(obj, Parameter):
                    added.add_parameter(obj.name, obj.param)

            # Remove the added component
            components.remove(comp)

            for child in list(comp.children.values()):
                add_comp_and_children(child, components, added)

        # Add component recursivly
        while components:
            add_comp_and_children(components[-1], components, ode)

        # Iterate over all components to add expressions
        for comp_name in self.all_expr_components_ordered:

            comp = self.all_components[comp_name]

            # If we should add component
            if comp not in collected_components:
                continue

            comp_comment = f"Expressions for the {comp.name} component"

            # Get corresponding component in new ODE
            added = old_new_map[comp]

            # Iterate over all objects of the component and save only
            # expressions and comments
            for obj in comp.ode_objects:

                # If saving an expression
                if isinstance(obj, Expression):

                    # The new sympy expression
                    new_expr = obj.expr.xreplace(subs)

                    # If the component is a Markov model
                    if comp.rates:

                        # Do not save State derivatives
                        if isinstance(obj, StateDerivative):
                            continue

                        # Save rate expressions slightly different
                        elif isinstance(obj, RateExpression):

                            states = obj.states
                            added._add_single_rate(
                                states[0].sym,
                                states[1].sym,
                                new_expr,
                            )
                            continue

                    # All other Expressions
                    setattr(added, str(obj), new_expr)

                # If saving a comment
                elif isinstance(obj, Comment) and str(obj) != comp_comment:
                    added.add_comment(str(obj))

        # Finalize ode and return it
        ode.finalize()
        return ode

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
            self._mass_matrix = sp.Matrix(
                N,
                N,
                lambda i, j: 1
                if i == j and isinstance(state_exprs[i], StateDerivative)
                else 0,
            )

        return self._mass_matrix

    @property
    def is_dae(self):
        """
        Return True if ODE is a DAE
        """
        if not self.is_complete:
            error("The ODE is not complete")

        return any(
            isinstance(expr, AlgebraicExpression) for expr in self.state_expressions
        )

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
            def_list += [sympycode(expr.expr) for expr in comp.intermediates]

            # Sort state expressions wrt stringified state names
            def_list += [
                sympycode(expr.expr)
                for expr in sorted(
                    self.state_expressions,
                    key=cmp_to_key(lambda o0, o1: cmp(str(o0.state), str(o1.state))),
                )
            ]

        h = hashlib.sha1()
        h.update(";".join(def_list).encode("utf-8"))
        return h.hexdigest()

    def _replace_object(self, old_obj, replaced_obj, replace_dicts):
        """
        Recursivley replace an expression by recreating the expression
        using passed replace dicts
        """

        # Iterate over expressions obj is used in and replace these
        self.present_ode_objects[old_obj.name] = replaced_obj
        replace_dicts[old_obj.sym] = replaced_obj.sym
        self.object_component[replaced_obj] = self.object_component.pop(old_obj)

        for old_expr in self.object_used_in[old_obj]:

            # Recreate expression
            replaced_expr = recreate_expression(old_expr, replace_dicts)

            # Update used_in dict
            self.object_used_in[replaced_obj].add(replaced_expr)

            # Update all object used in
            for dep in self.expression_dependencies[old_expr]:
                if isinstance(dep, Comment):
                    continue
                if old_expr in self.object_used_in[dep]:
                    self.object_used_in[dep].remove(old_expr)
                    self.object_used_in[dep].add(replaced_expr)

            # Exchange and update the dependencies
            if old_obj in self.expression_dependencies[old_expr]:
                self.expression_dependencies[old_expr].remove(old_obj)
            self.expression_dependencies[old_expr].add(replaced_obj)

            # FIXME: Do not remove the dependencies
            # self.expression_dependencies[updated_expr] = \
            #            self.expression_dependencies.pop(expr)
            self.expression_dependencies[replaced_expr] = self.expression_dependencies[
                old_expr
            ]

            # Find the index of old expression and exchange it with updated
            old_comp = self.object_component[old_expr]

            # Get indexed of old expr and overwrite it with new expr
            ind = old_comp.ode_objects.index(old_expr)
            old_comp.ode_objects[ind] = replaced_expr

            # Update the expressions this expression is used in
            self._replace_object(old_expr, replaced_expr, replace_dicts)

        # Remove information about the replaced objects
        self.object_used_in.pop(old_obj)

    def _handle_expr_component(self, comp, expr):
        """
        A help function to sort and add components in the ordered
        the intermediate expressions are added to the ODE
        """

        if len(self.all_expr_components_ordered) == 0:
            self.all_expr_components_ordered.append(comp.name)

            # Add a comment to the component
            comp.add_comment(f"Expressions for the {comp.name} component")

            # Recount the last added expression so the comment comes
            # infront of the expression
            expr._recount()

        # We are shifting expression components
        elif self.all_expr_components_ordered[-1] != comp.name:

            # Finalize the last component we visited
            self.all_components[
                self.all_expr_components_ordered[-1]
            ].finalize_component()

            # Append this component
            self.all_expr_components_ordered.append(comp.name)

            # Add a comment to the component
            comp.add_comment(f"Expressions for the {comp.name} component")

            # Recount the last added expression so the comment comes
            # infront of the expression
            expr._recount()

    def _expand_single_derivative(self, comp, obj, der_expr, replace_dict, dependent):
        """
        Expand a single derivative and register it as new derivative expression

        Returns True if an expression was actually added

        Populate replace dict with a replacement for the derivative if it is trivial
        """

        # Try accessing already registered derivative expressions
        der_expr_obj = self.present_ode_objects.get(sympycode(der_expr))

        # If excist continue
        if der_expr_obj:
            return False

        # Try expand the given derivative if it is directly expandable just
        # add a replacement for the derivative result
        der_result = der_expr.args[0].diff(der_expr.args[1])
        if not der_result.atoms(sp.Derivative):
            replace_dict[der_expr] = der_result
            return False

        if not isinstance(der_expr.args[0], AppliedUndef):
            error(
                "Can only register Derivatives of allready registered "
                "Expressions. Got: {0}".format(sympycode(der_expr.args[0])),
            )

        if not isinstance(der_expr.args[1], (AppliedUndef, sp.Symbol)):
            error(
                "Can only register Derivatives with a single dependent "
                "variabe. Got: {0}".format(sympycode(der_expr.args[1])),
            )

        # Get the expr and dependent variable objects
        expr_obj = self.present_ode_objects[sympycode(der_expr.args[0])]
        var_obj = self.present_ode_objects[sympycode(der_expr.args[1])]

        # If the dependent variable is time and the expression is a state
        # variable we raise an error as the user should already have created
        # the expression.
        if isinstance(expr_obj, State) and var_obj == self._time:
            error(
                "The expression {0} is dependent on the state "
                "derivative of {1} which is not registered in this ODE.".format(
                    obj,
                    expr_obj,
                ),
            )

        # If we get a Derivative(expr, t) we issue an error
        # if isinstance(expr_obj, Expression) and var_obj == self._time:
        #    error("All derivative expressions of registered expressions "\
        #          "need to be expanded with respect to time. Use "\
        #          "expr.diff(t) instead of Derivative(expr, t) ")

        if not isinstance(expr_obj, Expression):
            error(
                "Can only differentiate expressions or states. Got {0} as "
                "the derivative expression.".format(expr_obj),
            )

        # Expand derivative and see if it is trivial
        der_result = expr_obj.expr.diff(var_obj.sym)

        # If derivative result are trival we substitute it
        if (
            der_result.is_number
            or isinstance(der_result, (sp.Symbol, AppliedUndef))
            or (
                isinstance(der_result, (sp.Mul, sp.Pow, sp.Add))
                and len(der_result.args) == 2
                and all(
                    isinstance(arg, (sp.Number, sp.Symbol, AppliedUndef))
                    for arg in der_result.args
                )
            )
        ):
            replace_dict[der_expr] = der_result
            return False

        # Store expression
        comp.add_derivative(expr_obj, var_obj, der_result, dependent)

        return True
