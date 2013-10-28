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
from collections import deque
import re
import numpy as np

# Model parameters imports
from modelparameters.parameterdict import *
from modelparameters.codegeneration import ccode, cppcode, pythoncode, \
     sympycode, matlabcode

# Gotran imports
from gotran.common import check_arg, check_kwarg
from oderepresentation import ODERepresentation

__all__ = ["CodeGenerator", "CCodeGenerator", "CppCodeGenerator", \
           "MatlabCodeGenerator"]

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

class CodeGenerator(object):
    
    # Class attributes
    language = "python"
    line_ending = ""
    closure_start = ""
    closure_end = ""
    line_cont = "\\"
    comment = "#"
    index = lambda x, i : "[{0}]".format(i)
    indent = 4
    indent_str = " "
    max_line_length = 79
    to_code = lambda a,b,c,d="math" : pythoncode(b,c,d)

    def args(self, result_name):
        ret_args=[]
        if self.params.self_arg:
            ret_args.append("self")

        for arg in self.params.rhs_args:
            if arg == "s":
                ret_args.append("states")
            elif arg == "t":
                ret_args.append("time")
            elif arg == "p" and \
                 not self.oderepr.optimization.parameter_numerals:
                ret_args.append("parameters")

        ret_args.append("{0}=None".format(result_name))
        
        return ", ".join(ret_args)

    @staticmethod
    def default_params():
        
        return ParameterDict(rhs_args=OptionParam(\
            "stp", ["tsp", "stp", "spt"]),
                             self_arg = False)

    def __init__(self, oderepr, params=None):
        check_arg(oderepr, ODERepresentation, 0)

        params = params or {}

        self.params = self.default_params()
        self.params.update(params)

        self.oderepr = oderepr
        self.oderepr.update_index(self.index)

    def wrap_body_with_function_prototype(self, body_lines, name, args, \
                                          return_arg="", comment=""):
        """
        Wrap a passed body of lines with a function prototype
        """
        check_arg(body_lines, list)
        check_arg(name, str)
        check_arg(args, str)
        check_arg(return_arg, str)
        check_arg(comment, (str, list))
        
        prototype = ["def {0}({1}):".format(name, args)]
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

        if return_arg:
            body.append("return " + return_arg)

        # Append body to prototyp
        prototype.append(body)
        return prototype

    def _states_and_parameters_code(self):
        ode = self.oderepr.ode

        # Start building body
        body_lines = ["# Imports", "import numpy as np", "import math", \
                      "from math import pow, sqrt, log"]
        body_lines.append("")
        body_lines.append("# Assign states")
        body_lines.append("assert(len(states) == {0})".format(ode.num_states))
        if self.oderepr.optimization.use_state_names:
            body_lines.append(", ".join(state.name for i, state in \
                            enumerate(ode.states)) + " = states")
        
        # Add parameters code if not numerals
        if not self.oderepr.optimization.parameter_numerals:
            body_lines.append("")
            body_lines.append("# Assign parameters")
            body_lines.append("assert(len(parameters) == {0})".format(\
                ode.num_parameters))

            if self.oderepr.optimization.use_parameter_names:
                body_lines.append(", ".join(param.name for i, param in \
                        enumerate(ode.parameters)) + " = parameters")

        return body_lines

    def dy_body(self, result_name="dy"):
        """
        Generate body lines of code for evaluating state derivatives
        """

        from modelparameters.codegeneration import pythoncode

        ode = self.oderepr.ode

        body_lines = self._states_and_parameters_code()

        # Iterate over any body needed to define the dy
        for expr, name in self.oderepr.iter_dy_body():
            if name == "COMMENT":
                body_lines.append("")
                body_lines.append("# " + expr)
            else:
                body_lines.append(pythoncode(expr, name))

        # Init dy
        body_lines.append("")
        body_lines.append("# Init {0}".format(result_name))
        body_lines.append("if {0} is None:".format(result_name))
        body_lines.append(["{0} = np.zeros_like(states)".format(result_name)])
        
        # Add dy[i] lines
        for ind, (state, (derivative, expr)) in enumerate(\
            zip(ode.states, self.oderepr.iter_derivative_expr())):
            assert(state.sym == derivative[0].sym)
            body_lines.append(pythoncode(expr, "{0}[{1}]".format(result_name, ind)))

        # Return body lines 
        return body_lines
        
    def dy_code(self, indent=0, result_name="dy"):
        """
        Generate code for evaluating state derivatives
        """

        body_lines = self.dy_body(result_name)
        
        body_lines.append("")
        body_lines.append("# Return result")

        # Add function prototype
        dy_function = self.wrap_body_with_function_prototype(\
            body_lines, "rhs", self.args(result_name), \
            result_name, "Compute right hand side")
        
        return "\n".join(self.indent_and_split_lines(dy_function, indent=indent))

    def jacobian_body(self, result_name="jac"):
        """
        Generate body lines of code for evaluating state derivatives
        """

        from modelparameters.codegeneration import pythoncode

        ode = self.oderepr.ode

        body_lines = self._states_and_parameters_code()
        
        # Iterate over any body needed to define the jacobian
        for expr, name in self.oderepr.iter_jacobian_body():
            if name == "COMMENT":
                body_lines.append("")
                body_lines.append("# " + expr)
            else:
                body_lines.append(pythoncode(expr, name))

        # Init jacobian
        body_lines.append("")
        body_lines.append("# Init jacobian")
        body_lines.append("if {0} is None:".format(result_name))
        body_lines.append(["{0} = np.zeros((len(states), len(states)), "\
                           "dtype=np.float_)".format(result_name)])
        
        # Add jac[i,j] lines
        for (indi, indj), expr in self.oderepr.iter_jacobian_expr():
            body_lines.append(pythoncode(\
                expr, "{0}[{1}, {2}]".format(result_name, indi, indj)))

        # Return body lines 
        return body_lines

    def jacobian_code(self, indent=0, result_name="jac"):
        """
        Generate code for evaluating jacobian
        """

        body_lines = self.jacobian_body(result_name)
        
        body_lines.append("")
        body_lines.append("# Return jacobian")

        # Add function prototype
        jacobian_function = self.wrap_body_with_function_prototype(\
            body_lines, "jacobian", self.args(result_name), \
            return_arg=result_name, comment="Compute jacobian")
        
        return "\n".join(self.indent_and_split_lines(jacobian_function, indent=indent))
        
    def monitored_body(self, result_name="monitored"):
        """
        Generate body lines of code for evaluating state derivatives
        """

        from modelparameters.codegeneration import pythoncode

        ode = self.oderepr.ode

        # Start building body
        body_lines = ["# Imports", "import numpy as np", "import math", \
                      "from math import pow, sqrt, log"]
        body_lines.append("")
        body_lines.append("# Assign states")
        body_lines.append("assert(len(states) == {0})".format(ode.num_states))
        if self.oderepr.optimization.use_state_names:

            state_indices, state_names = [], []
            for ind, state in enumerate(ode.states):
                if state.name in self.oderepr.used_in_monitoring["states"]:
                    state_names.append(state.name)
                    state_indices.append("states[{0}]".format(ind))

            if state_names:
                body_lines.append(", ".join(state_names) + " = " +\
                                  ", ".join(state_indices))
        
        # Add parameters code if not numerals
        if not self.oderepr.optimization.parameter_numerals:
            body_lines.append("")
            body_lines.append("# Assign parameters")
            body_lines.append("assert(len(parameters) == {0})".format(\
                ode.num_parameters))

            if self.oderepr.optimization.use_parameter_names:
            
                parameter_indices, parameter_names = [], []
                for ind, param in enumerate(ode.parameters):
                    if param.name in self.oderepr.used_in_monitoring["parameters"]:
                        parameter_names.append(param.name)
                        parameter_indices.append("parameters[{0}]".format(ind))

                if parameter_names:
                    body_lines.append(", ".join(parameter_names) + " = " +\
                                      ", ".join(parameter_indices))

        # Iterate over any body needed to define the dy
        for expr, name in self.oderepr.iter_monitored_body():
            if name == "COMMENT":
                body_lines.append("")
                body_lines.append("# " + expr)
            else:
                body_lines.append(pythoncode(expr, name))

        # Init dy
        body_lines.append("")
        body_lines.append("# Init monitored")
        body_lines.append("if monitored is None:")
        body_lines.append(["monitored = np.zeros({0}, dtype=np.float_)".format(\
            ode.num_monitored_intermediates)])
        
        # Add monitored[i] lines
        ind = 0
        for monitored, expr in self.oderepr.iter_monitored_expr():
            if monitored == "COMMENT":
                body_lines.append("")
                body_lines.append("# " + expr)
            else:
                body_lines.append(pythoncode(expr, "monitored[{0}]".format(ind)))
                ind += 1

        # Return body lines 
        return body_lines

    def monitored_code(self, indent=0, result_name="monitored"):
        """
        Generate code for evaluating monitored variables
        """

        body_lines = self.monitored_body(result_name)
        
        body_lines.append("")
        body_lines.append("# Return monitored")

        # Add function prototype
        monitor_function = self.wrap_body_with_function_prototype(\
            body_lines, "monitor", self.args(result_name), \
            result_name, "Compute monitored intermediates")
        
        return "\n".join(self.indent_and_split_lines(monitor_function, \
                                                     indent=indent))

    def init_states_code(self, indent=0):
        """
        Generate code for setting initial condition
        """

        self_arg = self.params.self_arg

        # Start building body
        body_lines = ["# Imports", "import numpy as np",\
                      "from modelparameters.utils import Range", \
                      "", "# Init values"]
        body_lines.append("# {0}".format(", ".join("{0}={1}".format(\
            state.name, state.init) for state in \
                      self.oderepr.ode.states)))
        body_lines.append("init_values = np.array([{0}], dtype=np.float_)"\
                          .format(", ".join("{0}".format(\
                state.init if np.isscalar(state.init) else state.init[0])\
                            for state in self.oderepr.ode.states)))
        body_lines.append("")
        
        range_check = "lambda value : value {minop} {minvalue} and "\
                      "value {maxop} {maxvalue}"
        body_lines.append("# State indices and limit checker")

        body_lines.append("state_ind = dict({0})".format(\
            ", ".join("{0}=({1}, {2})".format(\
                state.param.name, i, repr(state.param._range))\
                for i, state in enumerate(self.oderepr.ode.states))))
        body_lines.append("")

        body_lines.append("for state_name, value in values.items():")
        body_lines.append(\
            ["if state_name not in state_ind:",
             ["raise ValueError(\"{{0}} is not a state.\".format(state_name))"],
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

        args = "self, **values" if self_arg else "**values"
        
        # Add function prototype
        init_function = self.wrap_body_with_function_prototype(\
            body_lines, "init_values", args, "init_values", \
            "Init values")
        
        return "\n".join(self.indent_and_split_lines(init_function, indent=indent))

    def init_param_code(self, indent=0):
        """
        Generate code for setting parameters
        """

        self_arg = self.params.self_arg

        # Start building body
        body_lines = ["# Imports", "import numpy as np",\
                      "from modelparameters.utils import Range", \
                      "", "# Param values"]
        body_lines.append("# {0}".format(", ".join("{0}={1}".format(\
            param.name, param.init) for param in \
                      self.oderepr.ode.parameters)))
        body_lines.append("param_values = np.array([{0}], dtype=np.float_)"\
                          .format(", ".join("{0}".format(\
                param.init if np.isscalar(param.init) else param.init[0]) \
                    for param in self.oderepr.ode.parameters)))
        body_lines.append("")
        
        range_check = "lambda value : value {minop} {minvalue} and "\
                      "value {maxop} {maxvalue}"
        body_lines.append("# Parameter indices and limit checker")

        body_lines.append("param_ind = dict({0})".format(\
            ", ".join("{0}=({1}, {2})".format(\
                state.param.name, i, repr(state.param._range))\
                for i, state in enumerate(\
                          self.oderepr.ode.parameters))))
        body_lines.append("")

        body_lines.append("for param_name, value in values.items():")
        body_lines.append(\
            ["if param_name not in param_ind:",
             ["raise ValueError(\"{{0}} is not a param\".format(param_name))"],
             # ["raise ValueError(\"{{0}} is not a param in the {0} ODE\"."\
             #  "format(param_name))".format(self.oderepr.name)],
             "ind, range = param_ind[param_name]",
             "if value not in range:",
             ["raise ValueError(\"While setting \'{0}\' {1}\".format("\
              "param_name, range.format_not_in(value)))"],
             "", "# Assign value",
             "param_values[ind] = value"])
            
        body_lines.append("")
        
        args = "self, **values" if self_arg else "**values"
        
        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "default_parameters", \
            args, "param_values", "Parameter values")
        
        return "\n".join(self.indent_and_split_lines(function, indent=indent))

    def state_name_to_index_code(self, indent=0):
        """
        Return code for index handling for states
        """
        self_arg = self.params.self_arg

        body_lines = []
        body_lines.append("state_inds = dict({0})".format(\
            ", ".join("{0}={1}".format(state.param.name, i) for i, state \
                      in enumerate(self.oderepr.ode.states))))
        body_lines.append("")
        body_lines.append("indices = []")
        body_lines.append("for state in states:")
        body_lines.append(\
            ["if state not in state_inds:",
             ["raise ValueError(\"Unknown state: '{0}'\".format(state))"],
             "indices.append(state_inds[state])"])
        body_lines.append("return indices if len(indices)>1 else indices[0]")

        args = "self, *states" if self_arg else "*states"
        
        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "state_indices", \
            args, "", "State indices")
        
        return "\n".join(self.indent_and_split_lines(function, indent=indent))

    def param_name_to_index_code(self, indent=0):
        """
        Return code for index handling for parameters
        """

        self_arg = self.params.self_arg

        body_lines = []
        body_lines.append("param_inds = dict({0})".format(\
            ", ".join("{0}={1}".format(param.param.name, i) for i, param \
                                        in enumerate(self.oderepr.ode.parameters))))
        body_lines.append("")
        body_lines.append("indices = []")
        body_lines.append("for param in params:")
        body_lines.append(\
            ["if param not in param_inds:",
             ["raise ValueError(\"Unknown param: '{0}'\".format(param))"],
             "indices.append(param_inds[param])"])
        body_lines.append("return indices if len(indices)>1 else indices[0]")

        args = "self, *params" if self_arg else "*params"
        
        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "param_indices", \
            args, "", "Param indices")
        
        return "\n".join(self.indent_and_split_lines(function, indent=indent))

    def code_list(self, indent=0):
        
        code = [self.init_param_code(indent),
                self.init_states_code(indent),
                self.dy_code(indent),
                self.state_name_to_index_code(indent), 
                self.param_name_to_index_code(indent),
                self.monitored_code(indent)
                ]

        if self.oderepr.optimization.generate_jacobian:
            code += [self.jacobian_code(indent)]

        return code

    def class_code(self):
        """
        Generate class code
        """

        name = self.oderepr.class_name

        code_list = self.code_list(indent=1)

        return  """
class {0}:

    @staticmethod
{1}
""".format(name, "\n\n    @staticmethod\n".join(code_list))

    def module_code(self):
        
        code_list = self.code_list()

        return  """# Gotran generated code for the  "{0}" model
from __future__ import division

{1}
""".format(self.oderepr.ode.name, "\n\n".join(code_list))
    

    @classmethod
    def indent_and_split_lines(cls, code_lines, indent=0, ret_lines=None, \
                               no_line_ending=False):
        """
        Combine a set of lines into a single string
        """

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

class CCodeGenerator(CodeGenerator):

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
    to_code = lambda a,b,c : ccode(b,c)
    
    def args(self, result_name=""):

        ret_args=[]
        for arg in self.params.rhs_args:
            if arg == "s":
                ret_args.append("const double* states")
            elif arg == "t":
                ret_args.append("double time")
            elif arg == "p" and \
                not self.oderepr.optimization.parameter_numerals \
                and self.params.parameters_in_signature:
                ret_args.append("const double* parameters")

        ret = ", ".join(ret_args)
        
        if result_name:
            ret += ", double* {0}".format(result_name)

        return  ret

    @staticmethod
    def default_params():
        
        return ParameterDict(rhs_args = OptionParam(\
            "stp", ["tsp", "stp", "spt"]),
                             parameters_in_signature = True)

    def wrap_body_with_function_prototype(self, body_lines, name, args, \
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
            prototype.append("{0} {1}".format(self.comment, comment))

        const = " const" if const else ""
        prototype.append("{0} {1}({2}){3}".format(return_type, name, args, const))

        # Append body to prototyp
        prototype.append(body_lines)
        return prototype
    
    def _states_and_parameters_code(self):

        ode = self.oderepr.ode
        
        # Start building body
        body_lines = ["", "// Assign states"]
        if self.oderepr.optimization.use_state_names:
            for i, state in enumerate(ode.states):
                state_name = state.name if state.name not in ["I"] \
                             else state.name + "_"
                body_lines.append("const double {0} = states[{1}]".format(\
                    state_name, i))
        
        # Add parameters code if not numerals
        if self.params.parameters_in_signature and \
               not self.oderepr.optimization.parameter_numerals:
            body_lines.append("")
            body_lines.append("// Assign parameters")

            if self.oderepr.optimization.use_parameter_names:
                for i, param in enumerate(ode.parameters):
                    param_name = param.name if param.name not in ["I"] \
                                 else param.name + "_"
                    body_lines.append("const double {0} = parameters[{1}]".\
                                      format(param_name, i))

        return body_lines

    def init_states_code(self, indent=0):
        """
        Generate code for setting initial condition
        """

        body_lines = []
        body_lines = ["values[{0}] = {1}; // {2}".format(\
            i, state.init if np.isscalar(state.init) else state.init[0], state.name)\
                      for i, state in enumerate(self.oderepr.ode.states)]

        # Add function prototype
        init_function = self.wrap_body_with_function_prototype(\
            body_lines, "init_values", "double* values", "", "Init values")
        
        return "\n".join(self.indent_and_split_lines(init_function, indent=indent))

    def init_param_code(self, indent=0):
        """
        Generate code for setting  parameters
        """

        body_lines = []
        body_lines = ["values[{0}] = {1}; // {2}".format(\
            i, param.init if np.isscalar(param.init) else param.init[0], param.name)\
                      for i, param in enumerate(self.oderepr.ode.parameters)]

        # Add function prototype
        init_function = self.wrap_body_with_function_prototype(\
            body_lines, "parameter_values", "double* values", "", \
            "Default parameter values")
        
        return "\n".join(self.indent_and_split_lines(init_function, indent=indent))

    def dy_body(self, result_name="dy"):
        """
        Generate body lines of code for evaluating state derivatives
        """
        
        ode = self.oderepr.ode

        body_lines = self._states_and_parameters_code()

        # Iterate over any body needed to define the dy
        declared_duplicates = []
        for expr, name in self.oderepr.iter_dy_body():

            name = str(name)
            
            if name == "COMMENT":
                body_lines.append("")
                body_lines.append("// " + expr)
            else:
                if name in ode._intermediate_duplicates:
                    if name not in declared_duplicates:
                        declared_duplicates.append(name)
                        if name in ["I"]:
                            name += "_"
                        name = "double " + name

                else:
                    if name in ["I"]:
                        name += "_"
                    name = "const double " + name

                body_lines.append(self.to_code(expr, name))

        # Add dy[i] lines
        for ind, (state, (derivative, expr)) in enumerate(\
            zip(ode.states, self.oderepr.iter_derivative_expr())):
            assert(state.sym == derivative[0].sym)
            body_lines.append(self.to_code(expr, "{0}[{1}]".format(result_name, ind)))

        body_lines.append("")
        
        # Return the body lines
        return body_lines
        
    def dy_code(self, indent=0, result_name="dy"):
        """
        Generate code for evaluating state derivatives
        """

        body_lines = self.dy_body(result_name)

        # Add function prototype
        dy_function = self.wrap_body_with_function_prototype(\
            body_lines, "rhs", self.args(result_name), \
            "", "Compute right hand side of {0}".format(self.oderepr.name))
        
        return "\n".join(self.indent_and_split_lines(\
            dy_function, indent=indent))

    def jacobian_body(self, result_name="jac"):

        ode = self.oderepr.ode

        body_lines = self._states_and_parameters_code()

        # Iterate over any body needed to define the dy
        declared_duplicates = []
        for expr, name in self.oderepr.iter_jacobian_body():

            name = str(name)
            
            if name == "COMMENT":
                body_lines.append("")
                body_lines.append("// " + expr)
            else:
                if name in ode._intermediate_duplicates:
                    if name not in declared_duplicates:
                        declared_duplicates.append(name)
                        if name in ["I"]:
                            name += "_"
                        name = "double " + name

                else:
                    if name in ["I"]:
                        name += "_"
                    name = "const double " + name

                body_lines.append(self.to_code(expr, name))

        # Add jac[i,j] lines
        num_states = ode.num_states
        for (indi, indj), expr in self.oderepr.iter_jacobian_expr():
            body_lines.append(self.to_code(expr, "{0}[{1}*{2}+{3}]".format(\
                result_name, indi, num_states, indj)))
 
        body_lines.append("")
        
        # Return the body lines
        return body_lines

    def jacobian_code(self, indent=0, result_name="jac"):
        """
        Generate code for evaluating state derivatives
        """

        body_lines = self.jacobian_body(result_name)

        # Add function prototype
        jacobian_function = self.wrap_body_with_function_prototype(\
            body_lines, "jacobian", self.args(result_name), \
            "", "Compute jacobian of {0}".format(self.oderepr.name))
        
        return "\n".join(self.indent_and_split_lines(\
            jacobian_function, indent=indent))

    def jacobian_action_body(self, result_name="jac_action"):

        ode = self.oderepr.ode

        body_lines = self._states_and_parameters_code()

        # Iterate over any body needed to define the dy
        declared_duplicates = []
        for expr, name in self.oderepr.iter_jacobian_action_body():

            name = str(name)
            
            if name == "COMMENT":
                body_lines.append("")
                body_lines.append("// " + expr)
            else:
                if name in ode._intermediate_duplicates:
                    if name not in declared_duplicates:
                        declared_duplicates.append(name)
                        if name in ["I"]:
                            name += "_"
                        name = "double " + name

                else:
                    if name in ["I"]:
                        name += "_"
                    name = "const double " + name

                body_lines.append(self.to_code(expr, name))

        # Add jac[i,j] lines
        num_states = ode.num_states
        for ind, expr in enumerate(self.oderepr.iter_jacobian_action_expr()):
            body_lines.append(self.to_code(expr, "{0}[{1}]".format(\
                result_name, ind)))
 
        body_lines.append("")
        
        # Return the body lines
        return body_lines

    def jacobian_action_code(self, indent=0, result_name="jac_action"):
        """
        Generate code for evaluating state derivatives
        """

        body_lines = self.jacobian_action_body(result_name)

        # Add function prototype
        jacobian_action_function = self.wrap_body_with_function_prototype(\
            body_lines, "jacobian_action", self.args(result_name), \
            "", "Compute jacobian action of {0}".format(self.oderepr.name))
        
        return "\n".join(self.indent_and_split_lines(\
            jacobian_action_function, indent=indent))

    def monitored_body(self, result_name="monitored"):
        """
        Generate body lines of code for evaluating monitored intermediates
        """

        ode = self.oderepr.ode

        # Start building body
        body_lines = ["", "// Assign states"]
        if self.oderepr.optimization.use_state_names:
            for ind, state in enumerate(ode.states):
                if state.name in self.oderepr.used_in_monitoring["states"]:
                    body_lines.append("const double {0} = states[{1}]".format(\
                        state.name, ind))
        
        # Add parameters code if not numerals
        if self.params.parameters_in_signature and \
               not self.oderepr.optimization.parameter_numerals:
            body_lines.append("")
            body_lines.append("// Assign parameters")

            if self.oderepr.optimization.use_parameter_names:
                for ind, param in enumerate(ode.parameters):
                    if param.name in self.oderepr.used_in_monitoring["parameters"]:
                        body_lines.append("const double {0} = parameters[{1}]".\
                                          format(param.name, ind))

        # Iterate over any body needed to define the monitored
        declared_duplicates = []
        for expr, name in self.oderepr.iter_monitored_body():

            name = str(name)
            
            if name == "COMMENT":
                body_lines.append("")
                body_lines.append("// " + expr)
            else:
                if name in ["I"]:
                    name += "_"
                name = "const double " + name
                body_lines.append(self.to_code(expr, name))

        # Add monitored[i] lines
        ind = 0 
        for monitored, expr in self.oderepr.iter_monitored_expr():
            if monitored == "COMMENT":
                body_lines.append("")
                body_lines.append("// " + expr)
            else:
                body_lines.append(self.to_code(expr, "{0}[{1}]".format(\
                    result_name, ind)))
                ind += 1

        body_lines.append("")
        
        # Return the body lines
        return body_lines
        
    def monitored_code(self, indent=0, result_name="monitored"):
        """
        Generate code for evaluating monitored intermediates
        """

        body_lines = self.monitored_body(result_name)

        # Add function prototype
        monitored_function = self.wrap_body_with_function_prototype(\
            body_lines, "monitor", self.args(result_name), \
            "", "Compute monitored intermediates {0}".format(self.oderepr.name))
        
        return "\n".join(self.indent_and_split_lines(\
            monitored_function, indent=indent))

    def dy_componentwise_body(self):
        oderepr = self.oderepr
        ode = oderepr.ode
        componentwise_dy_body = []
        for id, ((subs, expr), used) in enumerate(zip(\
            oderepr.iter_componentwise_dy(), oderepr.used_in_single_dy)):
            body_lines = []
            if oderepr.optimization.use_state_names:
                body_lines.append("// Assign states")
                for ind, state in enumerate(ode.states):
                    if state.name in used["states"]:
                        state_name = state.name if state.name not in ["I"] \
                                     else state.name + "_"
                        body_lines.append("const double {0} = states[{1}]".format(\
                            state_name, ind))

        
            # Add parameters code if not numerals
            if self.params.parameters_in_signature and \
                   not self.oderepr.optimization.parameter_numerals:

                if self.oderepr.optimization.use_parameter_names:
                    body_lines.append("")
                    body_lines.append("// Assign parameters")
                    for ind, param in enumerate(ode.parameters):
                        if param.name in used["parameters"]:
                            param_name = param.name if param.name not in ["I"] \
                                         else param.name + "_"
                            body_lines.append("const double {0} = parameters[{1}]".\
                                              format(param_name, ind))

            if subs:
                body_lines.append("")
                body_lines.append("// Common sub expressions for derivative {0}".format(id))
                
            for name, sub_expr in subs:
                if name in ["I"]:
                    name += "_"
                name = "const double " + str(name)
                body_lines.append(self.to_code(sub_expr, name))

            body_lines.append("")
            body_lines.append("// The expression")
            body_lines.append("return " + self.to_code(expr, None))
            body_lines.append("break")

            componentwise_dy_body.append("")
            componentwise_dy_body.append("// Component {0}".format(id))
            componentwise_dy_body.append("case {0}:".format(id))
            componentwise_dy_body.append(body_lines)

        body = ["// What component?"]
        body.append("switch (id)")
        componentwise_dy_body.append("")
        componentwise_dy_body.append("default:")
        default_lines = []
        if self.language == "C++":
            default_lines.append("throw std::runtime_error(\"Index out of bounds\")")
        else:
            default_lines.append("// Throw an exception...")
        default_lines.append("return 0.0")
        componentwise_dy_body.append(default_lines)
        body.append(componentwise_dy_body)
        
        return body


    def dy_componentwise_code(self, indent=0):

        body_lines = self.dy_componentwise_body()

        # Add function prototype
        args = "unsigned int id, " + self.args()
        componentwise_function = self.wrap_body_with_function_prototype(\
            body_lines, "componentwise", args, \
            "double", "Compute componentwise rhs {0}".format(self.oderepr.name))
        
        return "\n".join(self.indent_and_split_lines(\
            componentwise_function, indent=indent))

    def linearized_dy_body(self, result_name="values"):
        oderepr = self.oderepr
        ode = oderepr.ode

        # Start building body
        body_lines = ["", "// Assign states"]
        if self.oderepr.optimization.use_state_names:
            for ind, state in enumerate(ode.states):
                if state.name in self.oderepr.used_in_linear_dy["states"]:
                    state_name = state.name if state.name not in ["I"] \
                                 else state.name + "_"
                    body_lines.append("const double {0} = states[{1}]".format(\
                        state_name, ind))
        
        # Add parameters code if not numerals
        if self.params.parameters_in_signature and \
               not self.oderepr.optimization.parameter_numerals:
            body_lines.append("")
            body_lines.append("// Assign parameters")

            if self.oderepr.optimization.use_parameter_names:
                for ind, param in enumerate(ode.parameters):
                    if param.name in self.oderepr.used_in_linear_dy["parameters"]:
                        param_name = param.name if param.name not in ["I"] \
                                     else param.name + "_"
                        body_lines.append("const double {0} = parameters[{1}]".\
                                          format(param_name, ind))

        for expr, name in self.oderepr.iter_linerized_body():

            name = str(name)
            
            if name == "COMMENT":
                body_lines.append("")
                body_lines.append("// " + expr)
            else:
                if name in ["I"]:
                    name += "_"
                name = "const double " + name
                body_lines.append(self.to_code(expr, name))

        body_lines.append("")
        body_lines.append("// Linearized derivatives")

        for id, expr in self.oderepr.iter_linerized_expr():
            body_lines.append(self.to_code(expr, "{0}[{1}]".format(result_name, id)))

        return body_lines

    def linearized_dy_code(self, indent=0, result_name="values"):

        body_lines = self.linearized_dy_body(result_name)

        # Add function prototype
        linearized_function = self.wrap_body_with_function_prototype(\
            body_lines, "linearized_derivatives", self.args(result_name), \
            "", "Compute linearized derivatives {0}".format(self.oderepr.name))
        
        return "\n".join(self.indent_and_split_lines(\
            linearized_function, indent=indent))

    def code_list(self, indent=0):
        
        code = [self.init_param_code(indent),
                self.init_states_code(indent),
                self.dy_code(indent),
                self.monitored_code(indent),
                self.dy_componentwise_code(indent),
                self.linearized_dy_code(indent),
                ]

        if self.oderepr.optimization.generate_jacobian:
            code += [self.jacobian_code(indent)]

        return code

    def module_code(self):
        
        code_list = self.code_list()

        return  """// Gotran generated code for the "{0}" model

{1}
""".format(self.oderepr.ode.name, "\n\n".join(code_list))

class CppCodeGenerator(CCodeGenerator):
    
    language = "C++"

    # Class attributes
    to_code = lambda a,b,c : cppcode(b,c)

    def class_code(self):
        """
        Generate class code
        """

        name = self.oderepr.class_name

        code_list = self.code_list(indent=2)

        return  """
// Gotran generated class for the "{0}" model        
class {1}
{{

public:

{2}

}};
""".format(name, self.oderepr.class_name, "\n\n".join(code_list))

class MatlabCodeGenerator(CodeGenerator):
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
    to_code = lambda a,b,c : matlabcode(b,c)

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
        
        prototype = ["function {0} = {1}({2})".format(return_args, name, args)]
        body = []

        # Wrap comment if any
        if comment:
            if isinstance(comment, list):
                body.extend(comment)
            else:
                body.append(comment)

        # Extend the body with body lines
        body.extend(body_lines)

        # Append body to prototyp
        prototype.append(body)
        return prototype

    def default_value_code(self):
        """
        Create code for getting intital values and default parameter values
        """
        ode = self.oderepr.ode

        body_lines = []

        # Start building body
        body_lines.append("")
        body_lines.append("if nargout < 1 || nargout > 3")
        body_lines.append(["error('Expected 1-3 output arguments.')"])
        body_lines.append("")
        body_lines.append("% --- Default parameters values --- ")
        
        present_param_component = ""
        for param in ode.parameters:
            
            if present_param_component != param.component:
                present_param_component = param.component
                
                body_lines.append("")
                body_lines.append("% --- {0} ---".format(param.component))
            
            body_lines.append("params.{0} = {1}".format(\
                param.name, param.init if np.isscalar(param.init) else param.init[0]))

        body_lines.append("")
            
        # Default initial values and state names
        init_values = [""]
        init_values.append("% --- Default initial state values --- ")
        init_values.append("x0 = zeros({0}, 1)".format(ode.num_states))

        state_names = [""]
        state_names.append("% --- State names --- ")
        state_names.append("state_names = cell({0}, 1)".format(ode.num_states))
        
        present_state_component = ""
        for ind, state in enumerate(ode.states):
            
            if present_state_component != state.component:
                present_state_component = state.component

                init_values.append("")
                init_values.append("% --- {0} ---".format(state.component))
            
                state_names.append("")
                state_names.append("% --- {0} ---".format(state.component))

            init_values.append("x0({0}) = {1} % {2}".format(\
                ind + 1, state.init if np.isscalar(state.init) else state.init[0], \
                state.name))
            state_names.append("state_names{{{0}}} = \'{1}\'".format(ind + 1, state.name))

        init_values.append("varargout(1) = {x0}")
        state_names.append("varargout(2) = {state_names}")

        # Add bodys to code
        body_lines.append("if nargout == 2")
        body_lines.append(init_values)
        
        body_lines.append("")
        body_lines.append("if nargout == 3")
        body_lines.append(state_names)

        body_lines = self.wrap_body_with_function_prototype(\
            body_lines, "{0}_default".format(ode.name), "", "[params, varargout]",\
            ["% Default values for ODE model: {0}".format(ode.name),
             "% ------------------------------{0}".format(len(ode.name)*"-"),
             "%",
             "% params = {0}_default();".format(ode.name),
             "% [params, ic] = {0}_default();".format(ode.name),
             "% [params, ic, state_names] = {0}_default();".format(ode.name)])

        return "\n".join(self.indent_and_split_lines(body_lines))
    
    def dy_code(self):
        """
        Generate code for rhs evaluation for the ode
        """

        ode = self.oderepr.ode

        self.oderepr.set_parameter_prefix("p.")

        body_lines = [""]
        
        body_lines.append("if ~(nargin == 2 || nargin == 3)")
        body_lines.append(["error('Expected 2-3 input arguments.')"])
        body_lines.append("")
        body_lines.append("if nargin == 2")
        body_lines.append(["p = default_{0}()".format(ode.name)])
        body_lines.append("")

        body_lines.append("% --- State values --- ")

        present_state_component = ""
        for ind, state in enumerate(ode.states):
            
            if present_state_component != state.component:
                present_state_component = state.component
                
                body_lines.append("")
                body_lines.append("% --- {0} ---".format(state.component))
            
            body_lines.append("{0} = states({1})".format(state.name, ind+1))
        
        # Iterate over any body needed to define the dy
        for expr, name in self.oderepr.iter_dy_body():
            if name == "COMMENT":
                body_lines.append("")
                body_lines.append("% " + expr)
            else:
                body_lines.append(self.to_code(expr, name))

        # Add dy(i) lines
        body_lines.append("")
        body_lines.append("% Right hand side")
        body_lines.append("dy = zeros({0}, 1)".format(ode.num_states))
        for ind, (state, (derivative, expr)) in enumerate(\
            zip(ode.states, self.oderepr.iter_derivative_expr())):
            assert(state.sym == derivative[0].sym), "{0}!={1}".format(\
                state.sym, derivative[0].sym)
            body_lines.append(self.to_code(expr, "dy({0})".format(ind+1)))
        
        body_lines = self.wrap_body_with_function_prototype( \
            body_lines, ode.name, "time, states, p", "[dy]", \
            ["% {0}(time, states, varagin)".format(ode.name),
             "% ",
             "% Usage",
             "% -----",
             "% [p, x] = {0}_default();".format(ode.name),
             "% [T, S] = ode15s(@{0}, [0, 60], x, [], p);".format(ode.name),
             ])
        
        return "\n".join(self.indent_and_split_lines(body_lines))
        
