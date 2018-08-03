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

# System imports
from collections import deque, OrderedDict
import re
import types
import numpy as np

# Model parameters imports
from modelparameters.parameterdict import *
from modelparameters.codegeneration import ccode, cppcode, pythoncode, \
     sympycode, matlabcode

# Gotran imports
from gotran.common import check_arg, check_kwarg, error, warning
from gotran.common.options import parameters
from gotran.model.ode import ODE
from gotran.model.odeobjects import Comment, ODEObject
from gotran.model.expressions import Expression, Intermediate, \
     IndexedExpression, AlgebraicExpression
from gotran.codegeneration.codecomponent import CodeComponent
from gotran.codegeneration.algorithmcomponents import *
from gotran.codegeneration.solvercomponents import *
from functools import reduce

__all__ = ["PythonCodeGenerator", "CCodeGenerator", "CppCodeGenerator", \
           "MatlabCodeGenerator", "class_name", "CUDACodeGenerator"]

def class_name(name):
    check_arg(name, str)
    return name if name[0].isupper() else name[0].upper() + \
           (name[1:] if len(name) > 1 else "")

class BaseCodeGenerator(object):
    """
    Base class for all code generators
    """

    # Class attributes
    language = "None"
    line_ending = ""
    closure_start = ""
    closure_end = ""
    line_cont = "\\"
    comment = "#"
    index = lambda x, i : "[{0}]".format(i)
    indent = 4
    indent_str = " "
    max_line_length = 79
    to_code = lambda self,b,c,d : None
    float_types = dict(single="float32", double="float64")

    def __init__(self, params=None):

        params = params or {}

        self.params = self.default_parameters()
        self.params.update(params)

    @property
    def float_type(self):
        "Return the float type"
        return type(self).float_types[self.params.code.float_precision]

    @staticmethod
    def default_parameters():
        # Start out with a copy of the global parameters
        return parameters.generation.copy()

    def code_dict(self, ode,
                  monitored=None,
                  include_init=True,
                  include_index_map=True,
                  indent=0):
        """
        Generates a dict of code snippets

        Arguments
        ---------
        ode : ODE
            The ODE for which code will be generated
        monitored : list
            A list of name of monitored intermediates for which evaluation
            code will be generated.
        include_init : bool
            If True, code for initializing the states and parameters will
            be generated.
        include_index_map : bool
            If True, code for mapping a str to a index for the corresponding,
            state, parameters or monitored will be generated.
        indent : int
            The indentation level for the generated code
        """

        monitored = monitored or []
        check_arg(ode, ODE)
        check_kwarg(monitored, "monitored", list, itemtypes=str)
        functions = self.params.functions

        code = OrderedDict()

        # If generate init code
        if include_init:
            code["init_states"] = self.init_states_code(ode, indent)
            code["init_parameters"] = self.init_parameters_code(ode, indent)

        # If generate index map code
        if include_index_map:
            code["state_indices"] = self.state_name_to_index_code(ode, indent)
            code["parameter_indices"] = self.param_name_to_index_code(ode, indent)

        comps = []

        # Code for the right hand side evaluation?
        if functions.rhs.generate:
            comps.append(rhs_expressions(\
                ode, function_name=functions.rhs.function_name,
                result_name=functions.rhs.result_name,
                params=self.params.code))

        # Code for any monitored intermediates
        if monitored and functions.monitored.generate:
            if include_index_map:
                code["monitor_indices"] = self.monitor_name_to_index_code(\
                    ode, monitored, indent)

            comps.append(monitored_expressions(\
                ode, monitored,
                function_name=functions.monitored.function_name,
                result_name=functions.monitored.result_name,
                params=self.params.code))

        # Code for generation of the jacobian of the right hand side
        jac = None
        lu_fact = None
        if functions.jacobian.generate:
            jac = jacobian_expressions(\
                ode, function_name=functions.jacobian.function_name,
                result_name=functions.jacobian.result_name,
                params=self.params.code)

            comps.append(jac)

        # Code for the symbolic factorization of the jacobian
        if functions.lu_factorization.generate:

            if jac is None:
                jac = jacobian_expressions(\
                    ode, function_name=functions.jacobian.function_name,
                    result_name=functions.jacobian.result_name,
                    params=self.params.code)

            lu_fact = factorized_jacobian_expressions(\
                jac, function_name=functions.lu_factorization.function_name,
                params=self.params.code)

            comps.append(lu_fact)

        # Code for the forward backward substituion for a factorized jacobian
        if functions.forward_backward_subst.generate:

            if jac is None:
                jac = jacobian_expressions(\
                    ode, function_name=functions.jacobian.function_name,
                    result_name=functions.jacobian.result_name,
                    params=self.params.code)

            if lu_fact is None:

                lu_fact = factorized_jacobian_expressions(\
                    jac, function_name=functions.lu_factorization.function_name,
                    params=self.params.code)

            fb_subs_param = functions.forward_backward_subst
            fb_subst = forward_backward_subst_expressions(\
                lu_fact, function_name=fb_subs_param.function_name,
                result_name=fb_subs_param.result_name,
                residual_name=fb_subs_param.residual_name,
                params=self.params.code)

            comps.append(fb_subst)

        # Code for generation of linearized derivatives
        if functions.linearized_rhs_evaluation.generate:
            comps.append(linearized_derivatives(\
                ode, function_name=functions.linearized_rhs_evaluation.function_name,
                result_names=functions.linearized_rhs_evaluation.result_names,
                only_linear=functions.linearized_rhs_evaluation.only_linear,
                include_rhs=functions.linearized_rhs_evaluation.include_rhs,
                params=self.params.code))

        # Add code for solvers
        for solver, solver_params in list(self.params.solvers.items()):
            if solver_params.generate:
                kwargs = solver_params.copy(to_dict=True)
                kwargs.pop("generate")
                kwargs["params"] = self.params.code
                comps.append(eval(solver+"_solver")(\
                    ode, **kwargs))

        # Create code snippest of all
        code.update((comp.function_name, \
                     self.function_code(comp, indent=indent)) for comp in comps)

        if functions.componentwise_rhs_evaluation.generate:
            snippet = self.componentwise_code(ode, indent=indent)
            if snippet:
                code[functions.componentwise_rhs_evaluation.function_name] = \
                                                                snippet

        if ode.is_dae:
            mass = self.mass_matrix(ode, indent=indent)
            if mass is not None:
                code["mass_matrix"] = mass

        return code

    @classmethod
    def indent_and_split_lines(cls, code_lines, indent=0, ret_lines=None, \
                               no_line_ending=False):
        """
        Combine a set of lines into a single string
        """

        _re_str = re.compile(".*\"([\w\s]+)\".*")

        def _is_number(num_str):
            """
            A hack to check wether a str is a number
            """
            try:
                float(num_str)
                return True
            except:
                return False

        check_kwarg(indent, "indent", int, ge=0)
        ret_lines = ret_lines or []

        # Walk through the code_lines
        for line_ind, line in enumerate(code_lines):

            # If another closure is encountered
            if isinstance(line, list):

                # Add start closure sign if any
                if cls.closure_start:
                    ret_lines.append(cls.indent*indent*cls.indent_str + \
                                     cls.closure_start)

                ret_lines = cls.indent_and_split_lines(\
                    line, indent+1, ret_lines)

                # Add closure if any
                if cls.closure_end:
                    ret_lines.append(cls.indent*indent*cls.indent_str + \
                                     cls.closure_end)
                continue

            line_ending = "" if no_line_ending else cls.line_ending

            # Do not use line endings the line before and after a closure
            if line_ind + 1 < len(code_lines):
                if isinstance(code_lines[line_ind+1], list):
                    line_ending = ""

            # Check if we parse a comment line
            if len(line) > len(cls.comment) and cls.comment == \
               line[:len(cls.comment)]:
                is_comment = True
                line_ending = ""
            else:
                is_comment = False

            # Empty line
            if line == "":
                ret_lines.append(line)
                continue

            # Check for long lines
            if cls.indent*indent + len(line) + len(line_ending) > \
                   cls.max_line_length:

                # Divide along white spaces
                splitted_line = deque(line.split(" "))

                # If no split
                if splitted_line == line:
                    ret_lines.append("{0}{1}{2}".format(\
                        cls.indent*indent*cls.indent_str, line, \
                        line_ending))
                    continue

                first_line = True
                inside_str = False

                while splitted_line:
                    line_stump = []
                    indent_length = cls.indent*(indent if first_line or \
                                                 is_comment else indent + 1)
                    line_length = indent_length

                    # Check if we are not exeeding the max_line_length
                    # FIXME: Line continuation symbol is not included in
                    # FIXME: linelength
                    while splitted_line and \
                              (((line_length + len(splitted_line[0]) \
                                 + 1 + inside_str) < cls.max_line_length) \
                               or not (line_stump and line_stump[-1]) \
                               or _is_number(line_stump[-1][-1])):
                        line_stump.append(splitted_line.popleft())

                        # Add a \" char to first stub if inside str
                        if len(line_stump) == 1 and inside_str:
                            line_stump[-1] = "\""+line_stump[-1]

                        # Check if we get inside or leave a str
                        if not is_comment and ("\"" in line_stump[-1] and not \
                            ("\\\"" in line_stump[-1] or \
                             "\"\"\"" in line_stump[-1] or \
                                re.search(_re_str, line_stump[-1]))):
                            inside_str = not inside_str

                        # Check line length
                        line_length += len(line_stump[-1]) + 1 + \
                                (is_comment and not first_line)*(len(\
                            cls.comment) + 1)

                    # If we are inside a str and at the end of line add
                    if inside_str and not is_comment:
                        line_stump[-1] = line_stump[-1]+"\""

                    # Join line stump and add indentation
                    ret_lines.append(indent_length*cls.indent_str + \
                                     (is_comment and not first_line)* \
                                     (cls.comment+" ") + " ".join(line_stump))

                    # If it is the last line stump add line ending otherwise
                    # line continuation sign
                    ret_lines[-1] = ret_lines[-1] + (not is_comment)*\
                                    (cls.line_cont if splitted_line else \
                                     line_ending)

                    first_line = False
            else:
                ret_lines.append("{0}{1}{2}".format(\
                    cls.indent*indent*cls.indent_str, line, \
                    line_ending))

        return ret_lines

class PythonCodeGenerator(BaseCodeGenerator):

    # Class attributes
    language = "python"
    to_code = lambda self, expr, name : pythoncode(expr, name, self.ns)
    float_types = dict(single="float32", double="float_")

    def __init__(self, params=None, ns="math"):

        check_arg(ns, str)
        assert ns in ["math", "np", "numpy", "", "ufl"]
        self.ns = ns
        super(PythonCodeGenerator, self).__init__(params)

    def args(self, comp):
        """
        Build argument str
        """

        params = self.params.code
        default_arguments = params.default_arguments \
                            if comp.use_default_arguments else ""

        additional_arguments = comp.additional_arguments[:]

        skip_result = []
        ret_args = []
        for arg in default_arguments:
            if arg == "s":
                if params.states.array_name in comp.results:
                    skip_result.append(params.states.array_name)
                ret_args.append(params.states.array_name)

            elif arg == "t":
                if params.time.name in comp.results:
                    skip_result.append(params.time.name)
                ret_args.append(params.time.name)
                if "dt" in additional_arguments:
                    additional_arguments.remove("dt")
                    ret_args.append(params.dt.name)

            elif arg == "p" and params.parameters.representation != \
                     "numerals":
                if params.parameters.array_name in comp.results:
                    skip_result.append(params.parameters.array_name)
                ret_args.append(params.parameters.array_name)

        ret_args.extend(additional_arguments)

        # Arguments with default (None) values
        if params.body.in_signature and params.body.representation != "named":
            ret_args.append("{0}=None".format(params.body.array_name))

        for result_name in comp.results:
            if result_name not in skip_result:
                ret_args.append("{0}=None".format(result_name))

        return ", ".join(ret_args)

    def decorators(self):
        # FIXME: Make this extendable with mode decorators or make it possible
        # FIXME: to use other standard decorators like classmethod
        return "@staticmethod" if self.params.class_code else ""

    @staticmethod
    def wrap_body_with_function_prototype(body_lines, name, args, \
                                          comment="", decorators=""):
        """
        Wrap a passed body of lines with a function prototype
        """
        check_arg(body_lines, list)
        check_arg(name, str)
        check_arg(args, str)
        check_arg(comment, (str, list))
        check_arg(decorators, (str, list))

        prototype = []
        if decorators:
            if isinstance(decorators, list):
                prototype.extend(decorators)
            else:
                prototype.append(decorators)

        prototype.append("def {0}({1}):".format(name, args))
        body = []

        # Wrap comment if any
        if comment:
            body.append("\"\"\"")
            if isinstance(comment, list):
                body.extend(comment)
            else:
                body.append(comment)
            body.append("\"\"\"")

        # Extend the body with body lines
        body.extend(body_lines)

        # Append body to prototyp
        prototype.append(body)
        return prototype

    def _init_arguments(self, comp):

        check_arg(comp, CodeComponent)
        params = self.params.code

        default_arguments = params.default_arguments \
                            if comp.use_default_arguments else ""

        # Check if comp defines used_states if not use the root components
        # full_states attribute
        # FIXME: No need for full_states here...
        used_states = comp.used_states if hasattr(comp, "used_states") else \
                      comp.root.full_states

        used_parameters = comp.used_parameters if hasattr(comp, "used_parameters") else \
                          comp.root.parameters

        num_states = comp.root.num_full_states
        num_parameters = comp.root.num_parameters

        # Start building body
        body_lines = []
        if "s" in default_arguments and used_states:

            states_name = params.states.array_name
            body_lines.append("")
            body_lines.append("# Assign states")
            body_lines.append("assert(len({0}) == {1})".format(states_name, \
                                                               num_states))
            # Generate state assign code
            if params.states.representation == "named":

                # If all states are used
                if len(used_states) == len(comp.root.full_states):
                    body_lines.append(", ".join(\
                        state.name for i, state in enumerate(comp.root.full_states)) + \
                                      " = " + states_name)

                # If only a limited number of states are used
                else:
                    body_lines.append("; ".join(\
                        "{0}={1}[{2}]".format(state.name, states_name, ind) \
                        for ind, state in enumerate(comp.root.full_states) \
                        if state in used_states))

        # Add parameters code if not numerals
        if "p" in default_arguments and \
               params.parameters.representation in ["named", "array"] and \
               used_parameters:

            parameters_name = params.parameters.array_name
            body_lines.append("")
            body_lines.append("# Assign parameters")
            body_lines.append("assert(len({0}) == {1})".format(\
                        parameters_name, num_parameters))

            # Generate parameters assign code
            if params.parameters.representation == "named":

                # If all parameters are used
                if len(used_parameters) == len(comp.root.parameters):
                    body_lines.append(", ".join(\
                        param.name for i, param in enumerate(used_parameters)) + \
                                      " = " + parameters_name)

                # If only a limited number of states are used
                else:
                    body_lines.append("; ".join(\
                        "{0}={1}[{2}]".format(param.name, parameters_name, ind) \
                        for ind, param in enumerate(comp.root.parameters) \
                        if param in used_parameters))

        # If using an array for the body variables
        if params.body.representation != "named" and \
               params.body.array_name in comp.shapes:

            body_name = params.body.array_name
            body_lines.append("")
            body_lines.append("# Body array {0}".format(body_name))

            # If passing the body argument to the method
            if  params.body.in_signature:
                body_lines.append("if {0} is None:".format(body_name))
                body_lines.append(["{0} = np.zeros({1}, dtype=np.{2})".format(\
                    body_name, comp.shapes[body_name], self.float_type)])
                body_lines.append("else:".format(body_name))
                body_lines.append(["assert isinstance({0}, np.ndarray) and "\
                                   "{1}.shape=={2}".format(\
                            body_name, body_name, comp.shapes[body_name])])
            else:
                body_lines.append("{0} = np.zeros({1}, dtype=np.{2})".format(\
                    body_name, comp.shapes[body_name], self.float_type))

        # If initelizing results
        if comp.results:

            results = comp.results[:]

            if params.states.array_name in results:
                results.remove(params.states.array_name)
            if params.time.name in results:
                results.remove(params.time.name)
            if params.parameters.array_name in results:
                results.remove(params.parameters.array_name)

            if results:
                body_lines.append("")
                body_lines.append("# Init return args")

            for result_name in results:
                shape = comp.shapes[result_name]
                if len(shape) > 1:
                    if params.array.flatten:
                        shape = (reduce(lambda a,b:a*b, shape, 1),)

                body_lines.append("if {0} is None:".format(result_name))
                body_lines.append(["{0} = np.zeros({1}, dtype=np.{2})".format(\
                    result_name, shape, self.float_type)])
                body_lines.append("else:".format(result_name))
                body_lines.append(["assert isinstance({0}, np.ndarray) and "\
                "{1}.shape == {2}".format(result_name, result_name, shape)])

        return body_lines

    def function_code(self, comp, indent=0, include_signature=True):
        """
        Generate code for a single function given by a CodeComponent
        """

        check_arg(comp, CodeComponent)
        check_kwarg(indent, "indent", int)

        body_lines = ["# Imports", "import numpy as np"]
        if self.ns:
            body_lines.append("import {0}".format(self.ns))

        body_lines += self._init_arguments(comp)

        # Iterate over any body needed to define the dy
        for expr in comp.body_expressions:
            if isinstance(expr, Comment):
                body_lines.append("")
                body_lines.append("# " + str(expr))
            else:
                body_lines.append(self.to_code(expr.expr, expr.name))

        if comp.results:
            body_lines.append("")
            body_lines.append("# Return results")
            body_lines.append("return {0}".format(", ".join(\
                result_name if len(comp.shapes[result_name])>=1 and \
                comp.shapes[result_name][0]>1 else result_name+"[0]" \
                for result_name in comp.results)))

        if include_signature:

            # Add function prototype
            body_lines = self.wrap_body_with_function_prototype(\
                body_lines, comp.function_name, self.args(comp), \
                comp.description, self.decorators())

        return "\n".join(self.indent_and_split_lines(body_lines, indent=indent))

    def init_states_code(self, ode, indent=0):
        """
        Generate code for setting initial condition
        """

        check_arg(ode, ODE)

        # Get all full states
        states = ode.full_states

        # Start building body
        body_lines = ["# Imports", "import numpy as np",\
                      "from modelparameters.utils import Range",\
                      "", "# Init values"]
        body_lines.append("# {0}".format(", ".join("{0}={1}".format(\
            state.name, state.init) for state in states)))
        body_lines.append("init_values = np.array([{0}], dtype=np.{1})"\
                          .format(", ".join("{0}".format(state.init)\
                            for state in states), self.float_type))
        body_lines.append("")

        range_check = "lambda value : value {minop} {minvalue} and "\
                      "value {maxop} {maxvalue}"
        body_lines.append("# State indices and limit checker")

        body_lines.append("state_ind = dict([{0}])".format(\
            ", ".join("(\"{0}\",({1}, {2}))".format(\
                state.param.name, i, repr(state.param._range))\
                      for i, state in enumerate(states))))
        body_lines.append("")

        body_lines.append("for state_name, value in values.items():")
        body_lines.append(\
            ["if state_name not in state_ind:",
             ["raise ValueError(\"{0} is not a state.\".format(state_name))"],
             # FIXME: Outcommented because of bug in indent_and_split_lines
             # ["raise ValueError(\"{{0}} is not a state in the {0} ODE\"."\
             #"format(state_name))".format(self.oderepr.name)],
             "ind, range = state_ind[state_name]",
             "if value not in range:",
             ["raise ValueError(\"While setting \'{0}\' {1}\".format("\
              "state_name, range.format_not_in(value)))"],
             "", "# Assign value",
             "init_values[ind] = value"])

        body_lines.append("")
        body_lines.append("return init_values")

        # Add function prototype
        init_function = self.wrap_body_with_function_prototype(\
            body_lines, "init_state_values", "**values", \
            "Initialize state values", self.decorators())

        return "\n".join(self.indent_and_split_lines(init_function, indent=indent))

    def init_parameters_code(self, ode, indent=0):
        """
        Generate code for setting parameters
        """

        check_arg(ode, ODE)

        # Get all parameters
        parameters = ode.parameters

        # Start building body
        body_lines = ["# Imports", "import numpy as np",\
                      "from modelparameters.utils import Range",\
                      "", "# Param values"]
        body_lines.append("# {0}".format(", ".join("{0}={1}".format(\
            param.name, param.init) for param in parameters)))
        body_lines.append("init_values = np.array([{0}], dtype=np.{1})"\
                          .format(", ".join("{0}".format(param.init) \
                    for param in parameters), self.float_type))
        body_lines.append("")

        range_check = "lambda value : value {minop} {minvalue} and "\
                      "value {maxop} {maxvalue}"
        body_lines.append("# Parameter indices and limit checker")

        body_lines.append("param_ind = dict([{0}])".format(\
            ", ".join("(\"{0}\", ({1}, {2}))".format(\
                state.param.name, i, repr(state.param._range))\
                for i, state in enumerate(parameters))))
        body_lines.append("")

        body_lines.append("for param_name, value in values.items():")
        body_lines.append(\
            ["if param_name not in param_ind:",
             ["raise ValueError(\"{0} is not a parameter.\".format(param_name))"],
             # ["raise ValueError(\"{{0}} is not a param in the {0} ODE\"."\
             #  "format(param_name))".format(self.oderepr.name)],
             "ind, range = param_ind[param_name]",
             "if value not in range:",
             ["raise ValueError(\"While setting \'{0}\' {1}\".format("\
              "param_name, range.format_not_in(value)))"],
             "", "# Assign value",
             "init_values[ind] = value"])

        body_lines.append("")
        body_lines.append("return init_values")

        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "init_parameter_values", \
            "**values", "Initialize parameter values", self.decorators())

        return "\n".join(self.indent_and_split_lines(function, indent=indent))

    def state_name_to_index_code(self, ode, indent=0):
        """
        Return code for index handling for states
        """
        check_arg(ode, ODE)
        states = ode.full_states

        body_lines = []
        body_lines.append("state_inds = dict([{0}])".format(\
            ", ".join("(\"{0}\", {1})".format(state.param.name, i) for i, state \
                      in enumerate(states))))
        body_lines.append("")
        body_lines.append("indices = []")
        body_lines.append("for state in states:")
        body_lines.append(\
            ["if state not in state_inds:",
             ["raise ValueError(\"Unknown state: '{0}'\".format(state))"],
             "indices.append(state_inds[state])"])
        body_lines.append("if len(indices)>1:")
        body_lines.append(["return indices"])
        body_lines.append("else:")
        body_lines.append(["return indices[0]"])

        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "state_indices", \
            "*states", "State indices", self.decorators())

        return "\n".join(self.indent_and_split_lines(function, indent=indent))

    def param_name_to_index_code(self, ode, indent=0):
        """
        Return code for index handling for parameters
        """

        check_arg(ode, ODE)

        parameters = ode.parameters

        body_lines = []
        body_lines.append("param_inds = dict([{0}])".format(\
            ", ".join("(\"{0}\", {1})".format(param.param.name, i) for i, param \
                                        in enumerate(parameters))))
        body_lines.append("")
        body_lines.append("indices = []")
        body_lines.append("for param in params:")
        body_lines.append(\
            ["if param not in param_inds:",
             ["raise ValueError(\"Unknown param: '{0}'\".format(param))"],
             "indices.append(param_inds[param])"])
        body_lines.append("if len(indices)>1:")
        body_lines.append(["return indices"])
        body_lines.append("else:")
        body_lines.append(["return indices[0]"])

        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "parameter_indices", \
            "*params", "Parameter indices", self.decorators())

        return "\n".join(self.indent_and_split_lines(function, indent=indent))

    def monitor_name_to_index_code(self, ode, monitored, indent=0):
        """
        Return code for index handling for monitored
        """
        check_arg(ode, ODE)

        for expr_str in monitored:
            obj = ode.present_ode_objects.get(expr_str)
            if not isinstance(obj, Expression):
                error("{0} is not an intermediate or state expression in "\
                      "the {1} ODE".format(expr_str, ode))

        body_lines = []
        body_lines.append("monitor_inds = dict([{0}])".format(\
            ", ".join("(\"{0}\", {1})".format(monitor, i) \
                      for i, monitor in enumerate(monitored))))
        body_lines.append("")
        body_lines.append("indices = []")
        body_lines.append("for monitor in monitored:")
        body_lines.append(\
            ["if monitor not in monitor_inds:",
             ["raise ValueError(\"Unknown monitored: '{0}'\".format(monitor))"],
             "indices.append(monitor_inds[monitor])"])
        body_lines.append("if len(indices)>1:")
        body_lines.append(["return indices"])
        body_lines.append("else:")
        body_lines.append(["return indices[0]"])

        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "monitor_indices", \
            "*monitored", "Monitor indices", self.decorators())

        return "\n".join(self.indent_and_split_lines(function, indent=indent))

    def componentwise_code(self, ode, indent=0, include_signature=True, \
                           return_body_lines=False):
        warning("Generation of componentwise_rhs_evaluation code is not "
                "yet implemented for Python backend.")

    def mass_matrix(self, ode, indent=0):
        check_arg(ode, ODE)
        body_lines = ["", "import numpy as np",
                      "M = np.eye({0})".format(ode.num_full_states)]

        for ind, expr in enumerate(ode.state_expressions):
            if isinstance(expr, AlgebraicExpression):
                body_lines.append("M[{0},{0}] = 0".format(ind))

        body_lines.append("")
        body_lines.append("return M")

        body_lines = self.wrap_body_with_function_prototype(\
            body_lines, "mass_matrix", "", \
            "The mass matrix of the {0} ODE".format(ode.name))

        return "\n".join(self.indent_and_split_lines(body_lines, indent=indent))

    def class_code(self, ode, monitored=None):
        """
        Generate class code
        """

        # Force class code param to be True
        class_code_param = self.params.class_code
        self.params.class_code = True

        check_arg(ode, ODE)
        name = class_name(ode.name)
        code_list = list(self.code_dict(ode, monitored=monitored, indent=1).values())

        self.params.class_code = class_code_param
        return  """
class {0}:

{1}
""".format(name, "\n\n".join(code_list))

    def module_code(self, ode, monitored=None):

        # Force class code param to be False
        class_code_param = self.params.class_code
        self.params.class_code = False

        check_arg(ode, ODE)
        code_list = list(self.code_dict(ode, monitored).values())
        self.params.class_code = class_code_param

        return  """# Gotran generated code for the  "{0}" model
from __future__ import division

{1}
""".format(ode.name, "\n\n".join(code_list))

class CCodeGenerator(BaseCodeGenerator):

    # Class attributes
    language = "C"
    line_ending = ";"
    closure_start = "{"
    closure_end = "}"
    line_cont = ""
    comment = "//"
    index = lambda x, i : "[{0}]".format(i)
    indent = 2
    indent_str = " "
    to_code = lambda self, expr, name : ccode(expr, name, \
                                              self.params.code.float_precision)
    float_types = dict(single="float", double="double")

    def obj_name(self, obj):
        assert(isinstance(obj, ODEObject))
        return obj.name
        return obj.name if obj.name not in ["I"] else obj.name + "_"

    def args(self, comp):

        params = self.params.code
        default_arguments = params.default_arguments \
                            if comp.use_default_arguments else ""

        additional_arguments = comp.additional_arguments[:]

        skip_result = []
        ret_args = []
        for arg in default_arguments:
            if arg == "s":
                if params.states.array_name in comp.results:
                    skip_result.append(params.states.array_name)
                    ret_args.append("{0}* {1}".format(self.float_type, \
                                                      params.states.array_name))
                else:
                    ret_args.append("const {0}* {1}".format(self.float_type, \
                                                            params.states.array_name))
            elif arg == "t":
                if params.time.name in comp.results:
                    error("Cannot have the same name for the time argument as "\
                          "for a result argument.")
                ret_args.append("const {0} {1}".format(self.float_type, \
                                                       params.time.name))
                if "dt" in additional_arguments:
                    additional_arguments.remove("dt")
                    ret_args.append("const {0} {1}".format(self.float_type, \
                                                           params.dt.name))

            elif arg == "p" and params.parameters.representation != \
                     "numerals":
                if params.parameters.array_name in comp.results:
                    skip_result.append(params.parameters.array_name)
                    ret_args.append("{0}* {1}".format(\
                        self.float_type, params.parameters.array_name))
                else:
                    ret_args.append("const {0}* {1}".format(\
                        self.float_type, params.parameters.array_name))

                field_parameters = params["parameters"]["field_parameters"]

                # If empty
                # FIXME: Get rid of this by introducing a ListParam type in modelparameters
                if len(field_parameters) > 1 or \
                       (len(field_parameters)==1 and field_parameters[0] != ""):
                    if params.parameters.field_array_name in comp.results:
                        skip_result.append(params.parameters.field_array_name)
                        ret_args.append("{0}* {1}".format(\
                            self.float_type, params.parameters.field_array_name))
                    else:
                        ret_args.append("const {0}* {1}".format(\
                            self.float_type, params.parameters.field_array_name))


        ret_args.extend("{0}* {1}".format(self.float_type, arg) \
                        for arg in additional_arguments)

        if params.body.in_signature and params.body.representation != "named":
            ret_args.append("{0}* {1}".format(self.float_type,
                                              params.body.array_name))

        for result_name in comp.results:
            if result_name not in skip_result:
                ret_args.append("{0}* {1}".format(self.float_type, result_name))

        return ", ".join(ret_args)

    @classmethod
    def wrap_body_with_function_prototype(cls, body_lines, name, args, \
                                          return_type="", comment="", const=False):
        """
        Wrap a passed body of lines with a function prototype
        """
        check_arg(body_lines, list)
        check_arg(name, str)
        check_arg(args, str)
        check_arg(return_type, str)
        check_arg(comment, (str, list))

        return_type = return_type or "void"

        prototype = []
        if comment:
            prototype.append("// {0}".format(comment))

        const = " const" if const else ""
        prototype.append("{0} {1}({2}){3}".format(return_type, name, args, const))

        # Append body to prototyp
        prototype.append(body_lines)
        return prototype


    def _init_arguments(self, comp):

        check_arg(comp, CodeComponent)
        params = self.params.code
        default_arguments = params.default_arguments \
                            if comp.use_default_arguments else ""

        field_parameters = params["parameters"]["field_parameters"]

        # If empty
        # FIXME: Get rid of this by introducing a ListParam type in modelparameters
        if len(field_parameters) == 1 and field_parameters[0] == "":
            field_parameters = []

        state_offset = params["states"]["add_offset"]
        parameter_offset = params["parameters"]["add_offset"]
        field_parameter_offset = params["parameters"]["add_field_offset"]

        # Check if comp defines used_states if not use the root components
        # full_states attribute
        # FIXME: No need for full_states here...
        full_states = comp.root.full_states
        used_states = comp.used_states if hasattr(comp, "used_states") else \
                      full_states

        used_parameters = comp.used_parameters if hasattr(comp, "used_parameters") else \
                          comp.root.parameters
        all_parameters = comp.root.parameters

        field_parameters = [param for param in all_parameters \
                            if param.name in field_parameters]

        # Start building body
        body_lines = []
        def add_obj(obj, i, array_name, add_offset=False):
            offset = "{0}_offset + ".format(array_name) if add_offset else ""
            body_lines.append("const {0} {1} = {2}[{3}{4}]".format(\
                self.float_type, self.obj_name(obj), array_name, offset, i))

        if "s" in default_arguments and used_states:

            # Generate state assign code
            if params.states.representation == "named":

                states_name = params.states.array_name
                body_lines.append("")
                body_lines.append("// Assign states")

                # If all states are used
                for i, state in enumerate(full_states):
                    if state not in used_states:
                        continue
                    add_obj(state, i, states_name, state_offset)

        # Add parameters code if not numerals
        if "p" in default_arguments and \
               params.parameters.representation in ["named", "array"] and \
               used_parameters:

            # Generate parameters assign code
            if params.parameters.representation == "named":

                parameters_name = params.parameters.array_name
                body_lines.append("")
                body_lines.append("// Assign parameters")

                # If all states are used
                for i, param in enumerate(all_parameters):
                    if param not in used_parameters or \
                           param in field_parameters:
                        continue
                    add_obj(param, i, parameters_name, parameter_offset)

                field_parameters_name = params.parameters.field_array_name
                for i, param in enumerate(field_parameters):
                    if param not in used_parameters:
                        continue
                    add_obj(param, i, field_parameters_name,
                            field_parameter_offset)

        # If using an array for the body variables and b is not passed as argument
        if params.body.representation != "named" and \
               not params.body.in_signature and \
               params.body.array_name in comp.shapes:

            body_name = params.body.array_name
            body_lines.append("")
            body_lines.append("// Body array {0}".format(body_name))
            body_lines.append("{0} {1}[{2}]".format(self.float_type, body_name, \
                                                    comp.shapes[body_name][0]))

        return body_lines

    def init_states_code(self, ode, indent=0):
        """
        Generate code for setting initial condition
        """

        states_name = self.params.code.states.array_name
        offset = "{0}_offset + ".format(states_name) \
                 if self.params.code.states.add_offset else ""
        float_str = "" if self.params.code.float_precision == "double" else "f"
        body_lines = ["{0}[{1}{2}] = {3}{4}; // {5}".format(\
            states_name, offset, i, state.init, float_str, state.name) \
                      for i, state in enumerate(ode.full_states)]

        # Add function prototype
        init_function = self.wrap_body_with_function_prototype(\
            body_lines, "init_state_values", "{0}* {1}".format(\
                self.float_type, states_name), "", "Init state values")

        return "\n".join(self.indent_and_split_lines(init_function, indent=indent))

    def init_parameters_code(self, ode, indent=0):
        """
        Generate code for setting  parameters
        """

        parameter_name = self.params.code.parameters.array_name
        offset = "{0}_offset + ".format(parameter_name) \
                 if self.params.code.parameters.add_offset else ""
        float_str = "" if self.params.code.float_precision == "double" else "f"
        body_lines = []
        body_lines = ["{0}[{1}{2}] = {3}{4}; // {5}".format(\
            parameter_name, offset, i, param.init, float_str, param.name) \
                      for i, param in enumerate(ode.parameters)]

        # Add function prototype
        init_function = self.wrap_body_with_function_prototype(\
            body_lines, "init_parameters_values", "{0}* {1}".format(\
                self.float_type, parameter_name), "", "Default parameter values")

        return "\n".join(self.indent_and_split_lines(init_function, indent=indent))

    def state_name_to_index_code(self, ode, indent=0):
        """
        Return code for index handling for states
        """
        check_arg(ode, ODE)
        states = ode.full_states

        max_length = max(len(state.name) for state in states)

        body_lines = ["// State names"]
        body_lines.append("char names[][{0}] = {{{1}}}".format(\
            max_length+1, ", ".join("\"{0}\"".format(state.name) for state \
                                    in states)))
        body_lines.append("")
        body_lines.append("int i")
        body_lines.append("for (i=0; i<{0}; i++)".format(len(states)))
        body_lines.append(["if (strcmp(names[i], name)==0)",\
                           ["return i"]])
        body_lines.append("return -1")

        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "state_index", \
            "const char name[]", "int", "State index")

        return "\n".join(self.indent_and_split_lines(function, indent=indent))

    def param_name_to_index_code(self, ode, indent=0):
        """
        Return code for index handling for a parameter
        """
        check_arg(ode, ODE)
        parameters = ode.parameters

        max_length = max(len(param.name) for param in parameters)

        body_lines = ["// Parameter names"]
        body_lines.append("char names[][{0}] = {{{1}}}".format(\
            max_length + 1, ", ".join("\"{0}\"".format(param.name) for param \
                                      in parameters)))
        body_lines.append("")
        body_lines.append("int i")
        body_lines.append("for (i=0; i<{0}; i++)".format(len(parameters)))
        body_lines.append(["if (strcmp(names[i], name)==0)",\
                           ["return i"]])
        body_lines.append("return -1")

        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "parameter_index", \
            "const char name[]", "int", "Parameter index")

        return "\n".join(self.indent_and_split_lines(function, indent=indent))

    def monitor_name_to_index_code(self, ode, monitored, indent=0):
        """
        Return code for index handling for monitored
        """
        max_length = max(len(monitor) for monitor in monitored)

        body_lines = ["\\ Monitored names"]
        body_lines.append("char names[][{0}] = {{{1}}}".format(\
            max_length + 1, ", ".join("\"{0}\"".format(monitor) for monitor \
                                    in monitored)))
        body_lines.append("")
        body_lines.append("for (int i=0; i<{0}; i++)".format(len(parameters)))
        body_lines.append(["if (strcmp(names[i], name)==0)",\
                           ["return i"]])
        body_lines.append("return -1")

        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "monitored_index", \
            "const char name[]", "Monitor index")

        return "\n".join(self.indent_and_split_lines(function, indent=indent))

    def function_code(self, comp, indent=0, default_arguments=None, \
                      include_signature=True, return_body_lines=False):

        params = self.params.code
        default_arguments = default_arguments or params.default_arguments

        check_arg(comp, CodeComponent)
        check_kwarg(default_arguments, "default_arguments", str)
        check_kwarg(indent, "indent", int)

        body_lines = self._init_arguments(comp)

        # If named body representation we need to check for duplicates
        duplicates = set()
        declared_duplicates = set()
        if params.body.representation == "named":
            collected_names = set()
            for expr in comp.body_expressions:
                if isinstance(expr, Expression) and \
                       not isinstance(expr, IndexedExpression):
                    if expr.name in collected_names:
                        duplicates.add(expr.name)
                    else:
                        collected_names.add(expr.name)

        # Iterate over any body needed to define the dy
        for expr in comp.body_expressions:
            if isinstance(expr, Comment):
                body_lines.append("")
                body_lines.append("// " + str(expr))
                continue
            elif isinstance(expr, IndexedExpression):
                name = "{0}".format(self.obj_name(expr))
            elif expr.name in duplicates:
                if expr.name not in declared_duplicates:
                    name = "{0} {1}".format(self.float_type, \
                                            self.obj_name(expr))
                    declared_duplicates.add(expr.name)
                else:
                    name = "{0}".format(self.obj_name(expr))
            else:
                name = "const {0} {1}".format(self.float_type, \
                                              self.obj_name(expr))
            body_lines.append(self.to_code(expr.expr, name))

        if return_body_lines:
            return body_lines

        if include_signature:

            # Add function prototype
            body_lines = self.wrap_body_with_function_prototype(\
                body_lines, comp.function_name, self.args(comp), "", \
                comp.description)

        return "\n".join(self.indent_and_split_lines(body_lines, indent=indent))

    def componentwise_code(self, ode, indent=0, default_arguments=None, \
                      include_signature=True, return_body_lines=False):

        params = self.params.code
        default_arguments = default_arguments or params.default_arguments

        float_str = "" if params.float_precision == "double" else "f"

        # Create code for each individuate component
        body_lines = []
        body_lines.append("// Return value")
        body_lines.append("{0} dy_comp[1] = {{0.0{1}}}".format(self.float_type, float_str))
        body_lines.append("")
        body_lines.append("// What component?")
        body_lines.append("switch (id)")

        switch_lines = []
        for i, state in enumerate(ode.full_states):

            component_code = ["", "// Component {0} state {1}".format(\
                i, state.name), "case {0}:".format(i)]

            comp = componentwise_derivative(ode, i, params=params, result_name="dy_comp")
            component_code.append(self.function_code(comp, indent, \
                                                     default_arguments, \
                                                     include_signature=False, \
                                                     return_body_lines=True))
            component_code[-1].append("break")

            switch_lines.extend(component_code)

        default_lines = ["", "// Default", "default:"]
        if self.language == "C++":
            default_lines.append(["throw std::runtime_error(\"Index out of bounds\")"])
        else:
            default_lines.append(["// Throw an exception..."])

        switch_lines.extend(default_lines)
        body_lines.append(switch_lines)
        body_lines.append("")
        body_lines.append("// Return component")
        body_lines.append("return dy_comp[0]")

        if return_body_lines:
            return body_lines

        # Add function prototype
        if include_signature:
            body_lines = self.wrap_body_with_function_prototype(\
                body_lines, "rhs", "unsigned int id, " + self.args(comp), \
                self.float_type, "Evaluate componenttwise rhs of the ODE")

        return "\n".join(self.indent_and_split_lines(body_lines, indent=indent))

    def mass_matrix(self, ode, indent=0):
        warning("Generation of componentwise_rhs_evaluation code is not "
                "yet implemented for the C backend.")

    def module_code(self, ode, monitored=None):
        return  """// Gotran generated C/C++ code for the "{0}" model

{1}
""".format(ode.name, "\n\n".join(list(self.code_dict(ode, monitored=monitored).values())))

class CppCodeGenerator(CCodeGenerator):

    language = "C++"

    # Class attributes
    to_code = lambda self, expr, name : cppcode(\
        expr, name, self.params.code.float_precision)

    def class_code(self, ode, monitored=None):
        """
        Generate class code
        """

        return  """
// Gotran generated C++ class for the "{0}" model
class {1}
{{

public:

{2}

}};
""".format(ode.name, class_name(ode.name, monitored), "\n\n".join(\
                         list(self.code_dict(ode, indent=2).values())))

class CUDACodeGenerator(CCodeGenerator):
    # Class attributes
    language = "CUDA"

    @staticmethod
    def default_parameters():
        # Start out with a copy of the global parameters
        params = parameters.generation.copy()
        params.code.states.add_offset = True
        params.code.states.array_name = "d_states"
        params.code.parameters.add_field_offset = True
        params.code.parameters.array_name = "d_parameters"
        params.code.parameters.field_array_name = "d_field_parameters"
        return params

    @classmethod
    def wrap_body_with_function_prototype(cls, body_lines, name, args,
                                          return_type="", comment="",
                                          const=False, kernel=False, device=False):
        """
        Wrap a passed body of lines with a function prototype
        """
        check_arg(return_type, str)

        return_type = (return_type or "void")

        if kernel:
            return_type = '__global__ ' + return_type
        elif device:
            return_type = '__device__ ' + return_type

        # Call super class function wrapper
        return CCodeGenerator.wrap_body_with_function_prototype(\
            body_lines, name, args, return_type, comment, const)

    def init_states_code(self, ode, indent=0):
        """
        Generate code for setting initial condition
        """

        array_name = self.params.code.states.array_name
        float_str = "" if self.params.code.float_precision == "double" else "f"
        body_lines = ["const int thread_ind = blockIdx.x*blockDim.x + threadIdx.x"]
        n_nodes = self.params.code.n_nodes
        if n_nodes > 0:
            body_lines.append(
                "if (thread_ind >= {0}) return; "
                "// number of nodes exceeded".format(n_nodes))
        body_lines.append("const int {0}_offset = thread_ind*{1}".format(\
            array_name, ode.num_full_states))

        # Main body
        body_lines.extend("{0}[{0}_offset+{1}] = {2}{3}; // {4}".format(
                              array_name, i, state.init, float_str, state.name)
                          for i, state in enumerate(ode.full_states))

        # Add function prototype
        init_function = self.wrap_body_with_function_prototype(
            body_lines, "init_state_values", "{0} *{1}".format(\
                self.float_type, array_name),
            comment="Init state values", kernel=True)

        return "\n".join(self.indent_and_split_lines(init_function, indent=indent))

    def init_parameters_code(self, ode, indent=0):
        """
        Generate code for setting parameters
        """

        body_lines = []

        array_name = self.params.code.parameters.array_name

        # Main body
        float_str = "" if self.params.code.float_precision == "double" else "f"
        body_lines.extend(["{0}[{1}] = {2}{3}; // {4}".format(
            array_name, i, float_str, param.init, param.name)
                           for i, param in enumerate(ode.parameters)])

        # Add function prototype
        init_function = self.wrap_body_with_function_prototype(
            body_lines, "init_parameters_values", "{0} *{1}".format(\
                self.float_type, array_name),
            comment="Default parameter values", kernel=False)

        return "\n".join(self.indent_and_split_lines(init_function, indent=indent))

    def init_field_parameters_code(self, ode, indent=0):
        """
        Generate code for initialising field parameters
        """

        field_parameters = self.params.code.parameters.field_parameters

        # If empty
        # FIXME: Get rid of this by introducing a ListParam type in modelparameters
        if len(field_parameters) == 1 and field_parameters[0] == "":
            field_parameters = []

        array_name = self.params.code.parameters.array_name
        float_str = "" if self.params.code.float_precision == "double" else "f"
        base_array_name = array_name[2:] if array_name[:2] == "d_" else array_name
        field_array_name = "d_field_" + base_array_name

        parameters = ode.parameters
        parameter_names = [p.name for p in parameters]
        field_parameter_indices = [parameter_names.index(fp)
                                   for fp in field_parameters]
        num_field_parameters = len(field_parameters)

        body_lines = list()
        if num_field_parameters > 0:
            body_lines.append(
                "const int thread_ind = blockIdx.x*blockDim.x + threadIdx.x")
            n_nodes = self.params.code.n_nodes
            if n_nodes > 0:
                body_lines.append(
                    "if (thread_ind >= {0}) return; "
                    "// number of nodes exceeded".format(n_nodes))
            body_lines.append(
                "const int field_{0}_offset = thread_ind*{1}".format(
                    base_array_name, num_field_parameters))

        # Main body
        body_lines.extend(
            "{0}[field_{1}_offset + {2}] = {3}{4}; //{5}".format(
                field_array_name,
                base_array_name,
                i,
                parameters[field_parameter_indices[i]].init,
                float_str,
                field_parameter)
            for i, field_parameter in enumerate(field_parameters))

        # Add function prototype
        init_fparam_func = self.wrap_body_with_function_prototype(
            body_lines, "init_field_parameters",
            "{0} *{1}".format(self.float_type, field_array_name),
            comment="Initialize field parameters", kernel=True)

        return "\n".join(self.indent_and_split_lines(init_fparam_func,
                                                     indent=indent))

    def field_parameters_setter_code(self, ode, indent=0):
        # FIXME: Implement!
        # This is actually not needed
        return ""

    def field_states_getter_code(self, ode, indent=0):
        """
        Generate code for field state getter
        """

        field_states = self.params.code.states.field_states

        # If empty
        # FIXME: Get rid of this by introducing a ListParam type in modelparameters
        if len(field_states) == 1 and field_states[0] == "":
            field_states = []

        array_name = self.params.code.states.array_name
        base_array_name = array_name[2:] if array_name[:2] == "d_" else array_name
        field_array_name = "h_field_" + base_array_name

        states = ode.full_states
        # FIXME: Check that the state is really a state
        field_states = [state for state in states if state.name in field_states]

        num_field_states = len(field_states)
        array_name = self.params.code.states.array_name

        body_lines = []
        if num_field_states > 0:
            body_lines.append(
                "const int thread_ind = blockIdx.x*blockDim.x + threadIdx.x")
            n_nodes = self.params.code.n_nodes
            if n_nodes > 0:
                body_lines.append(
                    "if (thread_ind >= {0}) return; "
                    "// number of nodes exceeded".format(n_nodes))
            body_lines.append(
                "const int {0}_offset = thread_ind*{1}".format(
                    base_array_name, len(states)))
            body_lines.append(
                "const int field_{0}_offset = thread_ind*{1}".format(\
                    base_array_name, num_field_states))

        # Main body
        body_lines.extend(
            "{0}[field_{2}_offset + {3}] = "\
            "{1}[{2}_offset + {4}]; //{5}".format(\
                field_array_name, array_name,
                base_array_name, i, states.index(state), state.name)
            for i, state in enumerate(field_states))

        # Add function prototype
        getter_func = self.wrap_body_with_function_prototype(
            body_lines, "get_field_states",
            "const {0} *{1}, {0} *{2}".format(self.float_type, \
                                              array_name, field_array_name),
            comment="Get field states", kernel=True)

        return "\n".join(self.indent_and_split_lines(getter_func, indent=indent))

    def field_states_setter_code(self, ode, indent=0):
        """
        Generate code for field state setter
        """

        field_states = self.params.code.states.field_states

        # If empty
        # FIXME: Get rid of this by introducing a ListParam type in modelparameters
        if len(field_states) == 1 and field_states[0] == "":
            field_states = []

        array_name = self.params.code.states.array_name
        base_array_name = array_name[2:] if array_name[:2] == "d_" else array_name
        field_array_name = "h_field_" + base_array_name

        states = ode.full_states
        # FIXME: Check that the state is really a state
        field_states = [state for state in states if state.name in field_states]

        num_field_states = len(field_states)

        body_lines = []
        if num_field_states > 0:
            body_lines.append(
                "const int thread_ind = blockIdx.x*blockDim.x + threadIdx.x")
            n_nodes = self.params.code.n_nodes
            if n_nodes > 0:
                body_lines.append(
                    "if (thread_ind >= {0}) return; "
                    "// number of nodes exceeded".format(n_nodes))
            body_lines.append(
                "const int {0}_offset = thread_ind*{1}".format(
                    base_array_name, len(states)))
            body_lines.append(
                "const int field_{0}_offset = thread_ind*{1}".format(\
                    base_array_name, num_field_states))

        # Main body
        body_lines.extend(
            "{0}[{2}_offset + {3}] = "\
            "{1}[field_{2}_offset + {4}]; //{5}".format(\
                array_name, field_array_name,
                base_array_name, states.index(state), i, state.name)
            for i, state in enumerate(field_states))

        # Add function prototype
        setter_func = self.wrap_body_with_function_prototype(
            body_lines, "set_field_states",
            "const {0} *{1}, {0} *{2}".format(self.float_type, \
                                              field_array_name, array_name),
            comment="Set field states", kernel=True)

        return "\n".join(self.indent_and_split_lines(setter_func, indent=indent))

    def function_code(self, comp, indent=0, default_arguments=None, \
                      include_signature=True, return_body_lines=False):

        params = self.params.code
        field_parameters = params.parameters.field_parameters

        # If empty
        # FIXME: Get rid of this by introducing a ListParam type in modelparameters
        if len(field_parameters) == 1 and field_parameters[0] == "":
            field_parameters = []

        default_arguments = default_arguments or params.default_arguments

        check_arg(comp, CodeComponent)
        check_kwarg(default_arguments, "default_arguments", str)
        check_kwarg(indent, "indent", int)

        states_name = self.params.code.states.array_name
        field_parameter_name = self.params.code.parameters.field_array_name

        # Initialization
        body_lines = [
            "const int thread_ind = blockIdx.x*blockDim.x + threadIdx.x"]
        body_lines.append(
            "if (thread_ind >= n_nodes) return; "
            "// number of nodes exceeded")
        body_lines.append(
            "const int {0}_offset = thread_ind*{1}".format(\
                states_name, comp.root.num_full_states))

        if len(field_parameters) > 0:
            body_lines.append("const int {0}_offset = thread_ind*{1}".format(
                    field_parameter_name, len(field_parameters)))

        body_lines.extend(self._init_arguments(comp))

        # If named body representation we need to check for duplicates
        duplicates = set()
        declared_duplicates = set()
        if params.body.representation == "named":
            collected_names = set()
            for expr in comp.body_expressions:
                if isinstance(expr, Expression) and \
                       not isinstance(expr, IndexedExpression):
                    if expr.name in collected_names:
                        duplicates.add(expr.name)
                    else:
                        collected_names.add(expr.name)

        # Iterate over any body needed to define the dy
        for expr in comp.body_expressions:
            if isinstance(expr, Comment):
                body_lines.append("")
                body_lines.append("// " + str(expr))
                continue
            elif isinstance(expr, IndexedExpression):
                name = "{0}".format(self.obj_name(expr))
            elif expr.name in duplicates:
                if expr.name not in declared_duplicates:
                    name = "{0} {1}".format(self.float_type, \
                                            self.obj_name(expr))
                    declared_duplicates.add(expr.name)
                else:
                    name = "{0}".format(self.obj_name(expr))
            else:
                name = "const {0} {1}".format(self.float_type, \
                                              self.obj_name(expr))
            body_lines.append(self.to_code(expr.expr, name))

        if return_body_lines:
            return body_lines

        if include_signature:

            # Add function prototype
            body_lines = self.wrap_body_with_function_prototype(\
                body_lines, comp.function_name, self.args(comp) + ', const unsigned int n_nodes',
                "", \
                comp.description, kernel=True)

        return "\n".join(self.indent_and_split_lines(body_lines, indent=indent))

    def module_code(self, ode, monitored=None):

        code_list = list(self.code_dict(ode, monitored=monitored, \
                                   include_index_map=False).values())
        code_list.append(self.field_states_getter_code(ode))
        code_list.append(self.field_states_setter_code(ode))
        code_list.append(self.field_parameters_setter_code(ode))
        return  """// Gotran generated CUDA code for the "{0}" model

{1}
""".format(ode.name, "\n\n".join(code_list))

    def solver_code(self, ode, solver_type):
        code_list = list()
        code_list.append(self.function_code(
            get_solver_fn(solver_type)(ode, params=self.params.code)))
        code_list.append(self.init_states_code(ode))
        code_list.append(self.field_states_getter_code(ode))
        code_list.append(self.field_states_setter_code(ode))
        code_list.append(self.init_field_parameters_code(ode))
        code_list.append(self.field_parameters_setter_code(ode))
        return """// Gotran generated CUDA solver code for the "{0}" model

{1}
""".format(ode.name, "\n\n".join(code_list))

class MatlabCodeGenerator(BaseCodeGenerator):
    """
    A Matlab Code generator
    """

    # Class attributes
    language = "Matlab"
    line_ending = ";"
    closure_start = ""
    closure_end = "end"
    line_cont = "..."
    comment = "%"
    index = lambda x, i : "({0})".format(i)
    indent = 2
    indent_str = " "
    to_code = lambda self, expr, name : matlabcode(expr, name)

    @staticmethod
    def default_parameters():
        # Start out with a copy of the global parameters
        params = parameters.generation.copy()
        params.code.array.index_format = "()"
        params.code.array.index_offset = 1
        params.code.array.flatten = False

        return params

    def wrap_body_with_function_prototype(self, body_lines, name, args, \
                                          return_args="", comment=""):
        """
        Wrap a passed body of lines with a function prototype
        """
        check_arg(body_lines, list)
        check_arg(name, str)
        check_arg(args, str)
        check_arg(return_args, str)
        check_arg(comment, (str, list))

        if return_args:
            return_args = "[{0}] = ".format(return_args)

        prototype = ["function {0}{1}({2})".format(return_args, name, args)]
        body = []

        # Wrap comment if any
        if comment:
            if isinstance(comment, list):
                body.extend("% " + com for com in comment)
            else:
                body.append("% " + comment)

        # Extend the body with body lines
        body.extend(body_lines)

        # Append body to prototyp
        prototype.append(body)
        return prototype

    def args(self, comp):
        """
        Build argument str
        """

        params = self.params.code
        default_arguments = params.default_arguments \
                            if comp.use_default_arguments else ""

        input_args = []
        for arg in default_arguments:
            if arg == "s":
                input_args.append(params.states.array_name)
            elif arg == "t":
                input_args.append(params.time.name)
            elif arg == "p" and params.parameters.representation != \
                     "numerals":
                input_args.append(params.parameters.array_name)

        input_args.extend(comp.additional_arguments)

        # Arguments with default (None) values
        if params.body.in_signature and params.body.representation != "named":
            raise NotImplementedError()

        return ", ".join(comp.results), ", ".join(input_args)

    def code_dict(self, ode, monitored=None, include_init=True):

        code_dict = super(MatlabCodeGenerator, self).code_dict(\
            ode, monitored=monitored, include_init=include_init,\
            include_index_map=False)

        if monitored:
            code_dict["monitored_names"] = self.monitored_names_code(ode, monitored)

        return code_dict

    def init_parameters_code(self, ode, indent=0):
        """
        Create code for getting default parameter values
        """
        body_lines = []

        # Start building body
        body_lines.append("")
        body_lines.append("if nargout < 1 || nargout > 2")
        body_lines.append(["error('Expected 1-2 output arguments.')"])
        body_lines.append("")

        body_lines.append("% --- Default parameters values --- ")
        body_lines.append("parameters = zeros({0}, 1)".format(ode.num_parameters))

        parameter_names = [""]
        parameter_names.append("% --- Parameter names --- ")
        parameter_names.append("parameter_names = cell({0}, 1)".format(\
            ode.num_parameters))

        present_param_component = None
        for ind, param in enumerate(ode.parameters):

            if present_param_component != ode.object_component[param]:
                present_param_component = ode.object_component[param]

                body_lines.append("")
                body_lines.append("% --- {0} ---".format(present_param_component))

                parameter_names.append("")
                parameter_names.append("% --- {0} ---".format(present_param_component))

            body_lines.append("parameters({0}) = {1}; % {2}".format(\
                ind+1, param.init, param.name))
            parameter_names.append("parameter_names{{{0}}} = \'{1}\'".format(\
                ind + 1, param.name))

        parameter_names.append("varargout(1) = {parameter_names}")

        body_lines.append("")
        body_lines.append("if nargout == 2")
        body_lines.append(parameter_names)

        body_lines = self.wrap_body_with_function_prototype(\
            body_lines, "{0}_init_parameters".format(ode.name), "", \
            "parameters, varargout",\
            ["% Default parameter values for ODE model: {0}".format(ode.name),
             "% ----------------------------------------{0}".format(len(ode.name)*"-"),
             "%",
             "% parameters = {0}_init_parameters();".format(ode.name),
             "% [parameters, parameters_names] = "\
             "{0}_init_parameter();".format(ode.name)])

        return "\n".join(self.indent_and_split_lines(body_lines))

    def init_states_code(self, ode, indent=0):

        # Default initial values and state names
        body_lines = [""]
        body_lines.append("% --- Default initial state values --- ")
        body_lines.append("states = zeros({0}, 1)".format(ode.num_full_states))

        state_names = [""]
        state_names.append("% --- State names --- ")
        state_names.append("state_names = cell({0}, 1)".format(ode.num_full_states))

        present_state_component = None
        for ind, state in enumerate(ode.full_states):

            if present_state_component != ode.object_component[state]:
                present_state_component = ode.object_component[state]

                body_lines.append("")
                body_lines.append("% --- {0} ---".format(present_state_component))

                state_names.append("")
                state_names.append("% --- {0} ---".format(present_state_component))

            body_lines.append("states({0}) = {1}; % {2}".format(\
                ind + 1, state.init, state.name))
            state_names.append("state_names{{{0}}} = \'{1}\'".format(ind + 1, state.name))

        state_names.append("varargout(1) = {state_names}")

        # Add bodys to code
        body_lines.append("")
        body_lines.append("if nargout == 2")
        body_lines.append(state_names)

        body_lines = self.wrap_body_with_function_prototype(\
            body_lines, "{0}_init_states".format(ode.name), "", \
            "states, varargout",\
            ["% Default state values for ODE model: {0}".format(ode.name),
             "% ------------------------------------{0}".format(len(ode.name)*"-"),
             "%",
             "% states = {0}_init_states();".format(ode.name),
             "% [states, states_names] = {0}_init_states();".format(ode.name)])

        return "\n".join(self.indent_and_split_lines(body_lines))

    def monitored_names_code(self, ode, monitored):
        body_lines = [""]
        body_lines.append("% --- Monitored names --- ")
        body_lines.append("monitored_names = cell({0}, 1)".format(len(monitored)))

        present_monitor_component = None
        for ind, monitor in enumerate(monitored):

            obj = ode.present_ode_objects.get(monitor)
            component = ode.object_component[obj]
            if present_monitor_component != component:
                present_monitor_component = component

                body_lines.append("")
                body_lines.append("% --- {0} ---".format(component))

            body_lines.append("monitored_names{{{0}}} = \'{1}\'".format(\
                ind + 1, monitor))

        body_lines = self.wrap_body_with_function_prototype(\
            body_lines, "{0}_monitored_names".format(ode.name), "", \
            "monitored_names",\
            ["% Monitored value names for ODE model: {0}".format(ode.name),
             "% ---------------------- --------------{0}".format(len(ode.name)*"-"),
             "%",
             "% monitored_names = {0}_monitored_names();".format(ode.name)])

        return "\n".join(self.indent_and_split_lines(body_lines))

    def _init_arguments(self, comp):

        check_arg(comp, CodeComponent)
        params = self.params.code

        default_arguments = params.default_arguments \
                            if comp.use_default_arguments else ""

        # Check if comp defines used_states if not use the root components
        # full_states attribute
        # FIXME: No need for full_states here...
        used_states = comp.used_states if hasattr(comp, "used_states") else \
                      comp.root.full_states

        used_parameters = comp.used_parameters if hasattr(comp, "used_parameters") else \
                          comp.root.parameters

        num_states = comp.root.num_full_states
        num_parameters = comp.root.num_parameters

        # Start building body
        body_lines = []
        if "s" in default_arguments and used_states:

            states_name = params.states.array_name
            body_lines.append("")
            body_lines.append("% Assign states")
            body_lines.append("if length({0})~={1}".format(states_name, \
                                                           num_states))
            body_lines.append(["error('Expected the {0} array to be of "\
                               "size {1}.')".format(states_name, num_states)])

            # Generate state assign code
            if params.states.representation == "named":
                body_lines.append("; ".join(\
                    "{0}={1}({2})".format(state.name, states_name, ind+1) \
                    for ind, state in enumerate(comp.root.full_states) \
                    if state in used_states))

        # Add parameters code if not numerals
        if "p" in default_arguments and \
               params.parameters.representation in ["named", "array"] and \
               used_parameters:

            parameters_name = params.parameters.array_name
            body_lines.append("")
            body_lines.append("% Assign parameters")
            body_lines.append("if length({0})~={1}".format(parameters_name, \
                                                           num_parameters))
            body_lines.append(["error('Expected the {0} array to be of "\
                               "size {1}.')".format(parameters_name, num_parameters)])

            # Generate parameters assign code
            if params.parameters.representation == "named":

                body_lines.append("; ".join(\
                    "{0}={1}({2})".format(param.name, parameters_name, ind+1) \
                    for ind, param in enumerate(comp.root.parameters) \
                    if param in used_parameters))

        # If using an array for the body variables
        if params.body.representation != "named" and \
               params.body.array_name in comp.shapes:

            raise NotImplementedError("Using non-named representation of "\
                                      "the body arguments is not implemented.")
            #body_name = params.body.array_name
            #body_lines.append("")
            #body_lines.append("% Body array {0}".format(body_name))
            #
            ## If passing the body argument to the method
            #if  params.body.in_signature:
            #    body_lines.append("if {0} is None:".format(body_name))
            #    body_lines.append(["{0} = np.zeros({1}, dtype=np.{2})".format(\
            #        body_name, comp.shapes[body_name], self.float_type)])
            #    body_lines.append("else:".format(body_name))
            #    body_lines.append(["assert isinstance({0}, np.ndarray) and "\
            #                       "{1}.shape=={2}".format(\
            #                body_name, body_name, comp.shapes[body_name])])
            #else:
            #    body_lines.append("{0} = np.zeros({1}, dtype=np.{2})".format(\
            #        body_name, comp.shapes[body_name], self.float_type))

        # If initelizing results
        if comp.results:
            body_lines.append("")
            body_lines.append("% Init return args")

        for result_name in comp.results:
            shape = comp.shapes[result_name]
            if len(shape) > 1 and params.array.flatten:
                shape = (reduce(lambda a,b:a*b, shape, 1),)
            if len(shape) == 1:
                shape = (shape[0], 1)
            body_lines.append("{0} = zeros{1}".format(result_name, shape))

        return body_lines

    def function_code(self, comp, indent=0):

        check_arg(comp, CodeComponent)
        check_kwarg(indent, "indent", int)

        body_lines = self._init_arguments(comp)

        # Iterate over any body
        for expr in comp.body_expressions:
            if isinstance(expr, Comment):
                body_lines.append("")
                body_lines.append("% " + str(expr))
            else:
                body_lines.append(self.to_code(expr.expr, expr.name))

        # Add function prototype
        return_args, input_args = self.args(comp)
        body_lines = self.wrap_body_with_function_prototype(\
            body_lines, comp.root.name + "_" + comp.function_name, \
            input_args, return_args, comp.description)

        return "\n".join(self.indent_and_split_lines(body_lines, indent=indent))

    def mass_matrix(self, ode, indent=0):
        check_arg(ode, ODE)
        body_lines = ["", "M = eye({0})".format(ode.num_full_states)]

        for ind, expr in enumerate(ode.state_expressions):
            if isinstance(expr, AlgebraicExpression):
                body_lines.append("M({0},{0}) = 0".format(ind+1))

        body_lines = self.wrap_body_with_function_prototype(\
            body_lines, ode.name + "_mass_matrix", \
            "", "M", "The mass matrix of the {0} ODE".format(ode.name))

        return "\n".join(self.indent_and_split_lines(body_lines))
