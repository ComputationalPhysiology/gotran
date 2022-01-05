# Copyright (C) 2013 Johan Hake
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

__all__ = ["CodeComponent"]

import sys

# System imports
from collections import OrderedDict, defaultdict, deque
from functools import cmp_to_key

from modelparameters.codegeneration import sympycode
from modelparameters.logger import debug, error, info, warning

# ModelParameters imports
from modelparameters.sympytools import sp
from modelparameters.utils import Timer, check_arg, check_kwarg, tuplewrap
from modelparameters.sympy import cse
from modelparameters.sympy.core.function import AppliedUndef

from gotran.model.expressions import (
    Derivatives,
    Expression,
    IndexedExpression,
    Intermediate,
    ParameterIndexedExpression,
    StateDerivative,
    StateIndexedExpression,
    recreate_expression,
)
from gotran.model.ode import ODE
from gotran.model.odecomponent import ODEComponent
from gotran.model.odeobjects import (
    Comment,
    IndexedObject,
    Parameter,
    ParameterIndexedObject,
    State,
    StateIndexedObject,
    cmp,
)

# Local imports
from ..common import parameters


class CodeComponent(ODEComponent):
    """
    A wrapper class around an ODE. Its primary purpose is to help
    generate code.

    The class alows extraction and manipulation of the ODE expressions.
    """

    @staticmethod
    def default_parameters():
        """
        Return the default parameters for code generation
        """
        return parameters.generation.code.copy()

    def __init__(
        self,
        name,
        ode,
        function_name,
        description,
        params=None,
        use_default_arguments=True,
        additional_arguments=None,
        **results,
    ):
        """
        Creates a CodeComponent

        Arguments
        ---------
        name : str
            The name of the component. This str serves as the unique
            identifier of the Component.
        ode : gotran.ODE
            The parent component which need to be a ODE
        function_name : str
            The name of the generated function
        description : str
            A short description of what the code component are computing
        params : dict
            Parameters determining how the code should be generated
        use_default_arguments : bool
            If true state, time and parameters are expected to be used in the
            code component
        additional_arguments : list
            A list of str for additional arguments included in the signature.
        results : kwargs
            A dict of result expressions. The result expressions will
            be used to extract the body expressions for the code component.
        """
        params = params or {}
        additional_arguments = additional_arguments or []
        check_arg(ode, ODE)
        check_arg(name, str)
        check_arg(description, str)
        check_arg(function_name, str)
        check_kwarg(params, "params", dict)
        check_kwarg(use_default_arguments, "use_default_arguments", bool)
        check_kwarg(additional_arguments, "additional_arguments", list)

        super(CodeComponent, self).__init__(name, ode)

        # Turn off magic attributes, see ODEComponent.__setattr__
        # method
        self._allow_magic_attributes = False
        for result_name, result_expressions in list(results.items()):
            check_kwarg(
                result_expressions,
                result_name,
                list,
                itemtypes=(Expression, Comment),
            )

        # Store parameters
        self._params = self.default_parameters()
        self._params.update(params)

        # Store function name and description
        self.function_name = function_name
        self.description = description

        # Shapes for any indexed expressions or objects
        self.shapes = OrderedDict()

        # A map between expressions and recreated IndexedExpressions
        self.indexed_map = OrderedDict()

        # Init parameter or state replace dict
        if use_default_arguments:
            self.param_state_replace_dict = self._init_param_state_replace_dict()
        else:
            self.param_state_replace_dict = {}

        self.results = list(results.keys())
        # Recreate body expressions based on the given result_expressions
        if results:
            results, body_expressions = self._body_from_results(**results)
            self.body_expressions = self._recreate_body(body_expressions, **results)
        else:
            self.body_expressions = []

        # Store for later usage
        self.use_default_arguments = use_default_arguments
        self.additional_arguments = additional_arguments

    def add_indexed_expression(
        self,
        basename,
        indices,
        expr,
        add_offset=False,
        dependent=None,
        enum=None,
    ):
        """
        Add an indexed expression using a basename and the indices

        Arguments
        ---------
        basename : str
            The basename of the indexed expression
        indices : int, tuple of int
            The fixed indices identifying the expression
        expr : sympy.Basic, scalar
            The expression.
        add_offset : bool
            Add offset to indices
        dependent : gotran.ODEObject
            If given the count of this expression will follow as a
            fractional count based on the count of the dependent object
        """
        # Create an IndexedExpression in the present component
        timer = Timer("Add indexed expression")  # noqa: F841
        indices = tuplewrap(indices)

        # Check that provided indices fit with the registered shape
        if len(self.shapes[basename]) > len(indices):
            error(
                "Shape mismatch between indices {0} and registered "
                "shape for {1}{2}".format(indices, basename, self.shapes[basename]),
            )

        for dim, (index, shape_ind) in enumerate(zip(indices, self.shapes[basename])):
            if index >= shape_ind:
                error(
                    "Indices must be smaller or equal to the shape. Mismatch "
                    "in dim {0}: {1}>={2}".format(dim + 1, index, shape_ind),
                )

        # Create the indexed expression
        if enum is None:
            expr = IndexedExpression(
                basename,
                indices,
                expr,
                self.shapes[basename],
                self._params.array,
                add_offset,
                dependent,
                enum=enum,
            )
        elif isinstance(enum, State):
            state = enum
            expr = StateIndexedExpression(
                basename,
                indices,
                expr,
                state,
                self.shapes[basename],
                self._params.array,
                add_offset,
                dependent,
            )
        elif isinstance(enum, Parameter):
            parameter = enum
            expr = ParameterIndexedExpression(
                basename,
                indices,
                expr,
                parameter,
                self.shapes[basename],
                self._params.array,
                add_offset,
                dependent,
            )
        else:
            error(f"enum must be State or Parameter. Was {type(enum)}")
        self._register_component_object(expr, dependent)

        return expr.sym

    def add_indexed_object(self, basename, indices, add_offset=False):
        """
        Add an indexed object using a basename and the indices

        Arguments
        ---------
        basename : str
            The basename of the indexed expression
        indices : int, tuple of int
            The fixed indices identifying the expression
        add_offset : bool
            Add offset to indices
        """
        timer = Timer("Add indexed object")  # noqa: F841

        indices = tuplewrap(indices)

        # Check that provided indices fit with the registered shape
        if len(self.shapes[basename]) > len(indices):
            error(
                "Shape mismatch between indices {0} and registered "
                "shape for {1}{2}".format(indices, basename, self.shapes[basename]),
            )

        for dim, (index, shape_ind) in enumerate(zip(indices, self.shapes[basename])):
            if index >= shape_ind:
                error(
                    "Indices must be smaller or equal to the shape. Mismatch "
                    "in dim {0}: {1}>={2}".format(dim + 1, index, shape_ind),
                )

        # Create IndexedObject
        obj = IndexedObject(
            basename,
            indices,
            self.shapes[basename],
            self._params.array,
            add_offset,
        )
        self._register_component_object(obj)

        # Return the sympy version of the object
        return obj.sym

    def indexed_objects(self, *basenames):
        """
        Return a list of all indexed objects with the given basename,
        if no base names give all indexed objects are returned
        """
        if not basenames:
            basenames = list(self.shapes.keys())
        return [
            obj
            for obj in self.ode_objects
            if isinstance(obj, IndexedObject) and obj.basename in basenames
        ]

    def _init_param_state_replace_dict(self):
        """
        Create a parameter state replace dict based on the values in the
        global parameters
        """

        param_state_replace_dict = {}

        array_params = self._params["array"]
        param_repr = self._params["parameters"]["representation"]
        param_name = self._params["parameters"]["array_name"]
        param_offset = self._params["parameters"]["add_offset"]
        field_param_name = self._params["parameters"]["field_array_name"]
        field_param_offset = self._params["parameters"]["add_field_offset"]
        field_parameters = self._params["parameters"]["field_parameters"]

        # If empty
        # FIXME: Get rid of this by introducing a ListParam type in modelparameters
        if len(field_parameters) == 1 and field_parameters[0] == "":
            field_parameters = []

        for param in field_parameters:
            if not isinstance(self.root.present_ode_objects[param], Parameter):
                error(
                    f"Field parameter '{param}' is not a parameter in the '{self.root}'",
                )

        state_repr = self._params["states"]["representation"]
        state_name = self._params["states"]["array_name"]
        state_offset = self._params["states"]["add_offset"]
        time_name = self._params["time"]["name"]
        dt_name = self._params["dt"]["name"]

        # Create a map between states, parameters and the corresponding
        # IndexedObjects
        param_state_map = OrderedDict(
            [("states", OrderedDict()), ("parameters", OrderedDict())],
        )

        # Add states
        states = param_state_map["states"]
        for ind, state in enumerate(self.root.full_states):
            states[state] = StateIndexedObject(
                state_name,
                ind,
                state,
                (self.root.num_full_states,),
                array_params,
                state_offset,
            )

        # Add parameters
        parameters = param_state_map["parameters"]
        for ind, param in enumerate(self.root.parameters):

            if param.name in field_parameters:
                basename = field_param_name
                index = field_parameters.index(param.name)
                shape = (len(field_parameters),)
                offset = field_param_offset
            else:
                basename = param_name
                index = ind
                shape = (self.num_parameters,)
                offset = param_offset

            parameters[param] = ParameterIndexedObject(
                basename,
                index,
                param,
                shape,
                array_params,
                offset,
            )

        # If not having named parameters
        if param_repr == "numerals":
            param_state_replace_dict.update(
                (param.sym, param.init) for param in self.root.parameters
            )
        elif param_repr == "array":
            self.shapes[param_name] = (self.root.num_parameters,)
            if field_parameters:
                self.shapes["field_" + param_name] = (len(field_parameters),)
            param_state_replace_dict.update(
                (param.sym, indexed.sym)
                for param, indexed in list(param_state_map["parameters"].items())
            )

        if state_repr == "array":
            self.shapes[state_name] = (self.root.num_full_states,)
            param_state_replace_dict.update(
                (state.sym, indexed.sym)
                for state, indexed in list(param_state_map["states"].items())
            )

        param_state_replace_dict[self.root._time.sym] = sp.Symbol(time_name)
        param_state_replace_dict[self.root._dt.sym] = sp.Symbol(dt_name)

        self.indexed_map.update(param_state_map)

        # return  dicts
        return param_state_replace_dict

    def _body_from_results(self, **results):
        """
        Returns a sorted list of body expressions all used in the result expressions
        """

        # Store results
        self.results = list(results.keys())

        if not results:
            return {}, []

        # Check if we are using common sub expressions for body
        if self._params["body"]["use_cse"]:
            return self._body_from_cse(**results)
        else:
            return self._body_from_dependencies(**results)

    def _expanded_result_expressions(self, **results):

        # Extract all result expressions
        orig_result_expressions = sum(list(results.values()), [])

        # A map between result expression and result name
        result_names = dict(
            (result_expr, result_name)
            for result_name, result_exprs in list(results.items())
            for result_expr in result_exprs
        )

        # The expanded result expressions
        expanded_result_exprs = [
            self.root.expanded_expression(obj) for obj in orig_result_expressions
        ]

        # Set shape for result expressions
        for result_name, result_expressions in list(results.items()):
            if result_name not in self.shapes:
                self.shapes[result_name] = (len(result_expressions),)

        return orig_result_expressions, result_names, expanded_result_exprs

    def _only_result_expressions(self, **results):

        (
            orig_result_expressions,
            result_names,
            expanded_result_exprs,
        ) = self._expanded_result_expressions(**results)

    def _body_from_cse(self, **results):

        timer = Timer(f"Compute common sub expressions for {self.name}")  # noqa: F841

        (
            orig_result_expressions,
            result_names,
            expanded_result_exprs,
        ) = self._expanded_result_expressions(**results)

        state_offset = self._params["states"]["add_offset"]

        # Collect results and body_expressions
        body_expressions = []
        new_results = defaultdict(list)

        might_take_time = len(orig_result_expressions) >= 40

        if might_take_time:
            info(
                "Computing common sub expressions for {0}. Might take "
                "some time...".format(self.name),
            )
            sys.stdout.flush()

        # Call sympy common sub expression reduction
        cse_exprs, cse_result_exprs = cse(
            expanded_result_exprs,
            symbols=sp.numbered_symbols("cse_"),
            optimizations=[],
        )

        # Map the cse_expr to an OrderedDict
        cse_exprs = OrderedDict(cse_expr for cse_expr in cse_exprs)

        # Extract the symbols into a set for fast comparison
        cse_syms = set((sym for sym in cse_exprs))

        # Create maps between cse_expr and result expressions trying
        # to optimized the code by weaving in the result expressions
        # in between the cse_expr

        # A map between result expr and name and indices so we can
        # construct IndexedExpressions
        result_expr_map = defaultdict(list)

        # A map between last cse_expr used in a particular result expr
        # so that we can put the result expression right after the
        # last cse_expr it uses.
        last_cse_expr_used_in_result_expr = defaultdict(list)

        # Result expressions that does not contain any cse_sym
        result_expr_without_cse_syms = []

        # A map between cse_sym and its substitutes
        cse_subs = {}

        for ind, (orig_result_expr, result_expr) in enumerate(
            zip(orig_result_expressions, cse_result_exprs),
        ):

            # Collect information so that we can recreate the result
            # expression from
            result_expr_map[result_expr].append(
                (
                    result_names[orig_result_expr],
                    orig_result_expr.indices
                    if isinstance(orig_result_expr, IndexedExpression)
                    else ind,
                ),
            )

            # If result_expr does not contain any cse_sym
            if not any(cse_sym in cse_syms for cse_sym in result_expr.atoms()):
                result_expr_without_cse_syms.append(result_expr)

            else:

                # Get last cse_sym used in result expression
                last_cse_sym = sorted(
                    (cse_sym for cse_sym in result_expr.atoms() if cse_sym in cse_syms),
                    key=cmp_to_key(lambda a, b: cmp(int(a.name[4:]), int(b.name[4:]))),
                )[-1]

                if result_expr not in last_cse_expr_used_in_result_expr[last_cse_sym]:
                    last_cse_expr_used_in_result_expr[last_cse_sym].append(result_expr)

        debug(
            "Found {0} result expressions without any cse_syms.".format(
                len(result_expr_without_cse_syms),
            ),
        )

        # print ""
        # print "LAST cse_syms:", last_cse_expr_used_in_result_expr.keys()

        cse_cnt = 0
        atoms = [state.sym for state in self.root.full_states]
        atoms.extend(param.sym for param in self.root.parameters)

        # Collecte what states and parameters has been used
        used_states = set()
        used_parameters = set()

        self.add_comment(
            "Common sub expressions for the body and the " "result expressions",
        )
        body_expressions.append(self.ode_objects[-1])

        # Register the common sub expressions as Intermediates
        for cse_sym, expr in list(cse_exprs.items()):

            # print cse_sym, expr

            # If the expression is just one of the atoms of the ODE we
            # skip the cse expressions but add a subs for the atom We
            # also skip Relationals and Piecewise as the type checking
            # in Piecewise otherwise kicks in and destroys things for
            # us.
            if expr in atoms or isinstance(
                expr,
                (sp.Piecewise, sp.relational.Relational, sp.relational.Boolean),
            ):
                cse_subs[cse_sym] = expr.xreplace(cse_subs)
            else:
                # Add body expression as an intermediate expression
                sym = self.add_intermediate(f"cse_{cse_cnt}", expr.xreplace(cse_subs))
                obj = self.ode_objects.get(sympycode(sym))
                for dep in self.root.expression_dependencies[obj]:
                    if isinstance(dep, State):
                        used_states.add(dep)
                    elif isinstance(dep, Parameter):
                        used_parameters.add(dep)
                cse_subs[cse_sym] = sym
                cse_cnt += 1
                body_expressions.append(obj)

            # Check if we should add a result expressions
            if last_cse_expr_used_in_result_expr[cse_sym]:

                # Iterate over all registered result expr for this cse_sym
                for result_expr in last_cse_expr_used_in_result_expr.pop(cse_sym):

                    for result_name, indices in result_expr_map[result_expr]:

                        # Replace pure state and param expressions
                        # print cse_subs, result_expr
                        exp_expr = result_expr.xreplace(cse_subs)

                        sym = self.add_indexed_expression(
                            result_name,
                            indices,
                            exp_expr,
                            add_offset=state_offset,
                        )

                        expr = self.ode_objects.get(sympycode(sym))

                        for dep in self.root.expression_dependencies[expr]:
                            if isinstance(dep, State):
                                used_states.add(dep)
                            elif isinstance(dep, Parameter):
                                used_parameters.add(dep)

                        # Register the new result expression
                        new_results[result_name].append(expr)
                        body_expressions.append(expr)

        if might_take_time:
            info(" done")

        # Sort used state, parameters and expr
        self.used_states = sorted(used_states)
        self.used_parameters = sorted(used_parameters)

        return new_results, body_expressions

    def _body_from_dependencies(self, **results):

        timer = Timer(f"Compute dependencies for {self.name}")  # noqa: F841

        # Extract all result expressions
        result_expressions = sum(list(results.values()), [])

        # Check passed expressions
        ode_expr_deps = self.root.expression_dependencies
        exprs = set(result_expressions)
        not_checked = set()
        used_states = set()
        used_parameters = set()

        for expr in result_expressions:
            check_arg(
                expr,
                (Expression, Comment),
                context=CodeComponent._body_from_results,
            )

            if isinstance(expr, Comment):
                continue

            # Collect dependencies
            for obj in ode_expr_deps[expr]:
                if isinstance(obj, (Expression, Comment)):
                    not_checked.add(obj)
                elif isinstance(obj, State):
                    used_states.add(obj)
                elif isinstance(obj, Parameter):
                    used_parameters.add(obj)

        # use list to make the order consistent
        not_checked_list = sorted(not_checked)

        # Collect all dependencies
        while len(not_checked_list) > 0:
            dep_expr = not_checked_list.pop()
            exprs.add(dep_expr)
            for obj in ode_expr_deps[dep_expr]:
                if isinstance(obj, (Expression, Comment)):
                    if obj not in exprs:
                        if obj not in not_checked_list:
                            not_checked_list.append(obj)
                elif isinstance(obj, State):
                    used_states.add(obj)
                elif isinstance(obj, Parameter):
                    used_parameters.add(obj)

        # Sort used state, parameters and expr
        self.used_states = sorted(used_states)
        self.used_parameters = sorted(used_parameters)

        # Return a sorted list of all collected expressions
        return results, sorted(exprs)

    def _recreate_body(self, body_expressions, **results):
        """
        Create body expressions based on the given result_expressions

        In this method are all expressions replaced with something that should
        be used to generate code. The parameters in:

            parameters["generation"]["code"]

        decides how parameters, states, body expressions and indexed expressions
        are represented.

        """

        if not (results or body_expressions):
            return

        for result_name, result_expressions in list(results.items()):
            check_kwarg(
                result_expressions,
                result_name,
                list,
                context=CodeComponent._recreate_body,
                itemtypes=(Expression, Comment),
            )

        # Extract all result expressions
        result_expressions = sum(list(results.values()), [])

        # A map between result expression and result name
        result_names = dict(
            (result_expr, result_name)
            for result_name, result_exprs in list(results.items())
            for result_expr in result_exprs
        )

        timer = Timer(f"Recreate body expressions for {self.name}")  # noqa: F841

        # Initialize the replace_dictionaries
        replace_dict = self.param_state_replace_dict
        der_replace_dict = {}

        # Get a copy of the map of where objects are used in and their
        # present dependencies so any updates done in these dictionaries does not
        # affect the original dicts
        object_used_in = defaultdict(set)
        for expr, used in list(self.root.object_used_in.items()):
            object_used_in[expr].update(used)

        expression_dependencies = defaultdict(set)
        for expr, deps in list(self.root.expression_dependencies.items()):
            expression_dependencies[expr].update(deps)

        # Get body parameters
        body_repr = self._params["body"]["representation"]
        optimize_exprs = self._params["body"]["optimize_exprs"]

        # Set body related variables if the body should be represented by an array
        if "array" in body_repr:
            body_name = self._params["body"]["array_name"]
            available_indices = deque()
            max_index = -1
            body_ind = 0
            index_available_at = defaultdict(list)
            if body_name == result_name:
                error("body and result cannot have the same name.")

            # Initiate shapes with inf
            self.shapes[body_name] = (float("inf"),)

        # Iterate over body expressions and recreate the different expressions
        # according to state, parameters, body and result expressions
        replaced_expr_map = OrderedDict()
        new_body_expressions = []

        present_ode_objects = dict(
            (state.name, state) for state in self.root.full_states
        )
        present_ode_objects.update(
            (param.name, param) for param in self.root.parameters
        )
        old_present_ode_objects = present_ode_objects.copy()

        def store_expressions(expr, new_expr):
            "Help function to store new expressions"
            timer = Timer(  # noqa: F841
                f"Store expression while recreating body of {self.name}",
            )  # noqa: F841

            # Update sym replace dict
            if isinstance(expr, Derivatives):
                der_replace_dict[expr.sym] = new_expr.sym
            else:
                replace_dict[expr.sym] = new_expr.sym

            # Store the new expression for later references
            present_ode_objects[expr.name] = new_expr
            replaced_expr_map[expr] = new_expr

            # Append the new expression
            new_body_expressions.append(new_expr)

            # Update dependency information
            if expr in object_used_in:
                for dep in object_used_in[expr]:
                    if dep in expression_dependencies:
                        expression_dependencies[dep].remove(expr)
                        expression_dependencies[dep].add(new_expr)

                object_used_in[new_expr] = object_used_in.pop(expr)

            if expr in expression_dependencies:
                expression_dependencies[new_expr] = expression_dependencies.pop(expr)

        self.add_comment("Recreated body expressions")

        # The main iteration over all body_expressions
        for expr in body_expressions:

            # 1) Comments
            if isinstance(expr, Comment):
                new_body_expressions.append(expr)
                continue

            assert isinstance(expr, Expression)

            # 2) Check for expression optimizations
            if not (optimize_exprs == "none" or expr in result_expressions):

                timer_opt = Timer(  # noqa: F841
                    f"Handle expression optimization for {self.name}",
                )  # noqa: F841

                # If expr is just a number we exchange the expression with the
                # number
                if "numerals" in optimize_exprs and isinstance(expr.expr, sp.Number):
                    replace_dict[expr.sym] = expr.expr

                    # Remove information about this expr beeing used
                    for dep in object_used_in[expr]:
                        expression_dependencies[dep].remove(expr)
                    object_used_in.pop(expr)
                    continue

                # If the expr is just a symbol (symbol multiplied with a scalar)
                # we exchange the expression with the sympy expressions
                elif "symbols" in optimize_exprs and (
                    isinstance(expr.expr, (sp.Symbol, AppliedUndef))
                    or isinstance(expr.expr, sp.Mul)
                    and len(expr.expr.args) == 2
                    and isinstance(expr.expr.args[1], (sp.Symbol, AppliedUndef))
                    and expr.expr.args[0].is_number
                ):

                    # Add a replace rule based on the stored sympy expression
                    sympy_expr = expr.expr.xreplace(der_replace_dict).xreplace(
                        replace_dict,
                    )

                    if isinstance(expr.sym, sp.Derivative):
                        der_replace_dict[expr.sym] = sympy_expr
                    else:
                        replace_dict[expr.sym] = sympy_expr

                    # Get exchanged repr
                    if isinstance(expr.expr, (sp.Symbol, AppliedUndef)):
                        name = sympycode(expr.expr)
                    else:
                        name = sympycode(expr.expr.args[1])

                    dep_expr = present_ode_objects[name]

                    # If using reused body expressions we need to update the
                    # index information so that the index previously available
                    # for this expressions gets available at the last expressions
                    # the present expression is used in.
                    if (
                        isinstance(dep_expr, IndexedExpression)
                        and dep_expr.basename == body_name
                        and "reused" in body_repr
                    ):
                        ind = dep_expr.indices[0]

                        # Remove available index information
                        dep_used_in = sorted(object_used_in[dep_expr])
                        for used_expr in dep_used_in:
                            if ind in index_available_at[used_expr]:
                                index_available_at[used_expr].remove(ind)

                        # Update with new indices
                        all_used_in = object_used_in[expr].copy()
                        all_used_in.update(dep_used_in)

                        for used_expr in sorted(all_used_in, reverse=True):
                            if used_expr in body_expressions:
                                index_available_at[used_expr].append(ind)
                                break

                    # Update information about this expr beeing used
                    for dep in object_used_in[expr]:
                        expression_dependencies[dep].remove(expr)
                        expression_dependencies[dep].add(dep_expr)

                    object_used_in.pop(expr)
                    continue

                del timer_opt

            # 3) General operations for all Expressions that are kept

            # Before we process the expression we check if any indices gets
            # available with the expr (Only applies for the "reused" option for
            # body_repr.)
            if "reused" in body_repr:

                # Check if any indices are available at this expression ind
                available_indices.extend(index_available_at[expr])

            # Store a map of old name this will preserve the ordering of
            # expressions with the same name, similar to how this is treated in
            # the actual ODE.
            present_ode_objects[expr.name] = expr
            old_present_ode_objects[expr.name] = expr

            # 4) Handle result expression
            if expr in result_expressions:

                timer_result = Timer(  # noqa: F841
                    f"Handle result expressions for {self.name}",
                )  # noqa: F841

                # Get the result name
                result_name = result_names[expr]

                # If the expression is an IndexedExpression with the same basename
                # as the result name we just recreate it
                if (
                    isinstance(expr, IndexedExpression)
                    or isinstance(expr, StateIndexedExpression)
                    or isinstance(expr, ParameterIndexedExpression)
                ) and result_name == expr.basename:

                    new_expr = recreate_expression(expr, der_replace_dict, replace_dict)

                # Not an indexed expression
                else:

                    # Get index based on the original ordering
                    index = (results[result_name].index(expr),)
                    # Create the IndexedExpression
                    # NOTE: First replace any derivative expression replaces, then state and
                    # NOTE: params
                    if isinstance(expr, StateDerivative):
                        new_expr = StateIndexedExpression(
                            result_name,
                            index,
                            expr.expr.xreplace(der_replace_dict).xreplace(replace_dict),
                            expr.state,
                            (len(results[result_name]),),
                            array_params=self._params.array,
                        )
                    else:
                        new_expr = IndexedExpression(
                            result_name,
                            index,
                            expr.expr.xreplace(der_replace_dict).xreplace(replace_dict),
                            (len(results[result_name]),),
                            array_params=self._params.array,
                        )

                    if new_expr.basename not in self.indexed_map:
                        self.indexed_map[new_expr.basename] = OrderedDict()
                    self.indexed_map[new_expr.basename][expr] = new_expr

                    # Copy counter from old expression so it sort properly
                    new_expr._recount(expr._count)

                # Store the expressions
                store_expressions(expr, new_expr)

                del timer_result

            # 4) Handle indexed expression
            # All indexed expressions are just kept but recreated with updated
            # sympy expressions
            elif isinstance(expr, IndexedExpression):

                timer_indexed = Timer(  # noqa: F841
                    f"Handle indexed expressions for {self.name}",
                )  # noqa: F841

                new_expr = recreate_expression(expr, der_replace_dict, replace_dict)

                # Store the expressions
                store_expressions(expr, new_expr)

                del timer_indexed

            # 5) If replacing all body exressions with an indexed expression
            elif "array" in body_repr:

                timer_body = Timer(  # noqa: F841
                    f"Handle body expressions for {self.name}",
                )  # noqa: F841

                # 5a) If we reuse array indices
                if "reused" in body_repr:

                    if available_indices:
                        ind = available_indices.popleft()
                    else:
                        max_index += 1
                        ind = max_index

                    # Check when present ind gets available again
                    for used_expr in sorted(object_used_in[expr], reverse=True):
                        if used_expr in body_expressions:
                            index_available_at[used_expr].append(ind)
                            break
                    else:
                        warning("SHOULD NOT COME HERE!")

                # 5b) No reuse of array indices. Here each index corresponds to
                #     a distinct body expression
                else:

                    ind = body_ind

                    # Increase body_ind
                    body_ind += 1

                # Create the IndexedExpression
                new_expr = IndexedExpression(
                    body_name,
                    ind,
                    expr.expr.xreplace(der_replace_dict).xreplace(replace_dict),
                    array_params=self._params.array,
                    enum=expr.name,
                )

                if body_name not in self.indexed_map:
                    self.indexed_map[body_name] = OrderedDict()
                self.indexed_map[body_name][expr] = new_expr

                # Copy counter from old expression so they sort properly
                new_expr._recount(expr._count)

                # Store the expressions
                store_expressions(expr, new_expr)

                del timer_body

            # 6) If the expression is just an ordinary body expression and we
            #    are using named representation of body
            else:

                timer_expr = Timer(f"Handle expressions for {self.name}")  # noqa: F841
                # If the expression is a state derivative we need to add a
                # replacement for the Derivative symbol
                if isinstance(expr, StateDerivative):
                    new_expr = Intermediate(
                        sympycode(expr.sym),
                        expr.expr.xreplace(der_replace_dict).xreplace(replace_dict),
                    )
                    new_expr._recount(expr._count)
                else:
                    new_expr = recreate_expression(expr, der_replace_dict, replace_dict)

                del timer_expr

                # Store the expressions
                store_expressions(expr, new_expr)

        # Store indices for any added arrays
        if "reused_array" == body_repr:

            if max_index > -1:
                self.shapes[body_name] = (max_index + 1,)
            else:
                self.shapes.pop(body_name)

        elif "array" == body_repr:

            if body_ind > 0:
                self.shapes[body_name] = (body_ind,)
            else:
                self.shapes.pop(body_name)

        # Store the shape of the added result expressions
        for result_name, result_expressions in list(results.items()):
            if result_name not in self.shapes:
                self.shapes[result_name] = (len(result_expressions),)

        return new_body_expressions
