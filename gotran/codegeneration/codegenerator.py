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

# Model parameters imports
from modelparameters.parameters import *

# Gotran imports
from gotran.common import check_arg
from oderepresentation import ODERepresentation

_re_str = re.compile(".*\"([\w\s]+)\".*")

class CodeGenerator(object):
    def __init__(self, oderepr):
        check_arg(oderepr, ODERepresentation, 0)
        self.oderepr = oderepr
        self.max_line_length = 79
        self.init_language_specific_syntax()
        self.oderepr.update_index(self.index)

    def init_language_specific_syntax(self):
        self.language = "python"
        self.line_ending = ""
        self.closure_start = ""
        self.closure_end = ""
        self.line_cont = "\\"
        self.comment = "#"
        self.index = lambda i : "[{0}]".format(i)
        self.indent = 4
        self.indent_str = " "

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

    def dy_body(self):
        """
        Generate body lines of code for evaluating state derivatives
        """

        from modelparameters.codegeneration import pythoncode

        ode = self.oderepr.ode

        assert(not ode.is_dae)

        # Start building body
        body_lines = ["# Imports", "import numpy as np", "import math", \
                      "from math import pow, sqrt, log"]
        body_lines.append("")
        body_lines.append("# Assign states")
        body_lines.append("assert(len(states) == {0})".format(ode.num_states))
        if self.oderepr.optimization.use_state_names:
            body_lines.append(", ".join(state.name for i, state in \
                            enumerate(ode.iter_states())) + " = states")
        
        # Add parameters code if not numerals
        if not self.oderepr.optimization.parameter_numerals:
            body_lines.append("")
            body_lines.append("# Assign parameters")
            body_lines.append("assert(len(parameters) == {0})".format(\
                ode.num_parameters))

            if self.oderepr.optimization.use_parameter_names:
                body_lines.append(", ".join(param.name for i, param in \
                        enumerate(ode.iter_parameters())) + " = parameters")

        # Iterate over any body needed to define the dy
        for expr, name in self.oderepr.iter_dy_body():
            if name == "COMMENT":
                body_lines.append("")
                body_lines.append("# " + expr)
            else:
                body_lines.append(pythoncode(expr, name))

        # Init dy
        body_lines.append("")
        body_lines.append("# Init dy")
        body_lines.append("dy = np.zeros_like(states)")
        
        # Add dy[i] lines
        for ind, (state, (derivative, expr)) in enumerate(\
            zip(ode.iter_states(), self.oderepr.iter_derivative_expr())):
            assert(state.sym == derivative[0])
            body_lines.append(pythoncode(expr, "dy[{0}]".format(ind)))

        # Return body lines 
        return body_lines
        
    def dy_code(self):
        """
        Generate code for evaluating state derivatives
        """

        body_lines = self.dy_body()
        
        body_lines.append("")
        body_lines.append("# Return dy")

        # Add function prototype
        args = "time, states"
        if not self.oderepr.optimization.parameter_numerals:
            args += ", parameters"
        
        dy_function = self.wrap_body_with_function_prototype(\
            body_lines, "rhs", args, \
            "dy", "Calculate right hand side")
        
        return "\n".join(self.indent_and_split_lines(dy_function))


    def monitored_body(self):
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
            for ind, state in enumerate(ode.iter_states()):
                if state.name in self.oderepr._used_in_monitoring["states"]:
                    state_names.append(state.name)
                    state_indices.append("states[{0}]".format(ind))
            
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
                for ind, param in enumerate(ode.iter_parameters()):
                    if param.name in self.oderepr._used_in_monitoring["parameters"]:
                        parameter_names.append(param.name)
                        parameter_indices.append("parameters[{0}]".format(ind))
            
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
        body_lines.append("monitored = np.zeros({0}, dtype=np.float_)".format(\
            ode.num_monitored_intermediates))
        
        # Add monitored[i] lines
        for ind, (monitored, expr) in enumerate(\
            self.oderepr.iter_monitored_expr()):
            if monitored == "COMMENT":
                body_lines.append("")
                body_lines.append("# " + expr)
            else:
                body_lines.append(pythoncode(expr, "monitored[{0}]".format(ind)))

        # Return body lines 
        return body_lines


    def monitored_code(self):
        """
        Generate code for evaluating monitored variables
        """

        body_lines = self.monitored_body()
        
        body_lines.append("")
        body_lines.append("# Return monitored")

        # Add function prototype
        args = "time, states"
        if not self.oderepr.optimization.parameter_numerals:
            args += ", parameters"
        
        monitor_function = self.wrap_body_with_function_prototype(\
            body_lines, "monitor", args, \
            "monitored", "Calculate monitored intermediates")
        
        return "\n".join(self.indent_and_split_lines(monitor_function))

    def init_states_code(self):
        """
        Generate code for setting initial condition
        """

        # Start building body
        body_lines = ["# Imports", "import numpy as np",\
                      "from modelparameters.utils import Range", \
                      "", "# Init values"]
        body_lines.append("# {0}".format(", ".join("{0}={1}".format(\
            state.name, state.init) for state in \
                      self.oderepr.ode.iter_states())))
        body_lines.append("init_values = np.array([{0}], dtype=np.float_)"\
                          .format(", ".join("{0}".format(\
                state.init if np.isscalar(state.init) else state.init[0])\
                            for state in self.oderepr.ode.iter_states())))
        body_lines.append("")
        
        range_check = "lambda value : value {minop} {minvalue} and "\
                      "value {maxop} {maxvalue}"
        body_lines.append("# State indices and limit checker")

        body_lines.append("state_ind = dict({0})".format(\
            ", ".join("{0}=({1}, {2})".format(\
                state.param.name, i, repr(state.param._range))\
                for i, state in enumerate(self.oderepr.ode.iter_states()))))
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
        
        # Add function prototype
        init_function = self.wrap_body_with_function_prototype(\
            body_lines, "init_values", "**values", "init_values", \
            "Init values")
        
        return "\n".join(self.indent_and_split_lines(init_function))

    def init_param_code(self):
        """
        Generate code for setting parameters
        """

        # Start building body
        body_lines = ["# Imports", "import numpy as np",\
                      "from modelparameters.utils import Range", \
                      "", "# Param values"]
        body_lines.append("# {0}".format(", ".join("{0}={1}".format(\
            param.name, param.init) for param in \
                      self.oderepr.ode.iter_parameters())))
        body_lines.append("param_values = np.array([{0}], dtype=np.float_)"\
                          .format(", ".join("{0}".format(param.init) \
                    for param in self.oderepr.ode.iter_parameters())))
        body_lines.append("")
        
        range_check = "lambda value : value {minop} {minvalue} and "\
                      "value {maxop} {maxvalue}"
        body_lines.append("# Parameter indices and limit checker")

        body_lines.append("state_ind = dict({0})".format(\
            ", ".join("{0}=({1}, {2})".format(\
                state.param.name, i, repr(state.param._range))\
                for i, state in enumerate(\
                          self.oderepr.ode.iter_parameters()))))
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
        
        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "default_parameters", \
            "**values", "param_values", "Parameter values")
        
        return "\n".join(self.indent_and_split_lines(function))

    def indent_and_split_lines(self, code_lines, indent=0, ret_lines=None, \
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
                if self.closure_start:
                    ret_lines.append(self.indent*indent*self.indent_str + \
                                     self.closure_start)
                
                ret_lines = self.indent_and_split_lines(\
                    line, indent+1, ret_lines)
                
                # Add closure if any
                if self.closure_end:
                    ret_lines.append(self.indent*indent*self.indent_str + \
                                     self.closure_end)
                continue
            
            line_ending = "" if no_line_ending else self.line_ending
            # Do not use line endings the line before and after a closure
            if line_ind + 1 < len(code_lines):
                if isinstance(code_lines[line_ind+1], list):
                    line_ending = ""

            # Check of we parse a comment line
            if len(line) > len(self.comment) and self.comment == \
               line[:len(self.comment)]:
                is_comment = True
                line_ending = ""
            else:
                is_comment = False
                
            # Empty line
            if line == "":
                ret_lines.append(line)
                continue

            # Check for long lines
            if self.indent*indent + len(line) + len(line_ending) > \
                   self.max_line_length:

                # Divide along white spaces
                splitted_line = deque(line.split(" "))

                # If no split
                if splitted_line == line:
                    ret_lines.append("{0}{1}{2}".format(\
                        self.indent*indent*self.indent_str, line, \
                        line_ending))
                    continue
                    
                first_line = True
                inside_str = False
                
                while splitted_line:
                    line_stump = []
                    indent_length = self.indent*(indent if first_line or \
                                                 is_comment else indent + 1)
                    line_length = indent_length

                    # FIXME: Line continuation symbol is not included in
                    # FIXME: linelength
                    while splitted_line and \
                              (((line_length + len(splitted_line[0]) \
                                 + 1 + inside_str) < self.max_line_length) \
                               or not line_stump):
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
                            self.comment) + 1)

                    # If we are inside a str and at the end of line add
                    if inside_str and not is_comment:
                        line_stump[-1] = line_stump[-1]+"\""
                        
                    # Join line stump and add indentation
                    ret_lines.append(indent_length*self.indent_str + \
                                     (is_comment and not first_line)* \
                                     (self.comment+" ") + " ".join(line_stump))

                    # If it is the last line stump add line ending otherwise
                    # line continuation sign
                    ret_lines[-1] = ret_lines[-1] + (not is_comment)*\
                                    (self.line_cont if splitted_line else \
                                     line_ending)
                
                    first_line = False
            else:
                ret_lines.append("{0}{1}{2}".format(\
                    self.indent*indent*self.indent_str, line, \
                    line_ending))

        return ret_lines

class CCodeGenerator(CodeGenerator):
    def init_language_specific_syntax(self):
        from modelparameters.codegeneration import ccode

        self.language = "C"
        self.line_ending = ";"
        self.closure_start = "{"
        self.closure_end = "}"
        self.line_cont = ""
        self.comment = "//"
        self.index = lambda i : "[{0}]".format(i)
        self.indent = 2
        self.indent_str = " "
        self.to_code = ccode
    
    def wrap_body_with_function_prototype(self, body_lines, name, args, \
                                          return_type="", comment=""):
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

        prototype.append("{0} {1}({2})".format(return_type, name, args))
        body = []

        # Append body to prototyp
        prototype.append(body_lines)
        return prototype
    
    def dy_body(self, parameters_in_signature=False, result_name="dy"):
        """
        Generate body lines of code for evaluating state derivatives
        """
        
        ode = self.oderepr.ode

        assert(not ode.is_dae)

        # Start building body
        body_lines = ["", "// Assign states"]
        if self.oderepr.optimization.use_state_names:
            for i, state in enumerate(ode.iter_states()):
                body_lines.append("const double {0} = states[{1}]".format(\
                    state.name, i))
        
        # Add parameters code if not numerals
        if parameters_in_signature and \
               not self.oderepr.optimization.parameter_numerals:
            body_lines.append("")
            body_lines.append("// Assign parameters")

            if self.oderepr.optimization.use_parameter_names:
                for i, param in enumerate(ode.iter_parameters()):
                    body_lines.append("const double {0} = parameters[{1}]".\
                                      format(param.name, i))

        # Iterate over any body needed to define the dy
        declared_duplicates = []
        for expr, name in self.oderepr.iter_dy_body():

            name = str(name)
            
            if name == "COMMENT":
                body_lines.append("")
                body_lines.append("// " + expr)
            else:
                if name in ode._intermediates_duplicates:
                    if name not in declared_duplicates:
                        declared_duplicates.append(name)
                        name = "double " + name

                else:
                    name = "const double " + name

                body_lines.append(self.to_code(expr, name))

        # Add dy[i] lines
        for ind, (state, (derivative, expr)) in enumerate(\
            zip(ode.iter_states(), self.oderepr.iter_derivative_expr())):
            assert(state.sym == derivative[0])
            body_lines.append(self.to_code(expr, "{0}[{1}]".format(result_name, ind)))

        body_lines.append("")
        
        # Return the body lines
        return body_lines
        
    def dy_code(self, parameters_in_signature=False, result_name="dy"):
        """
        Generate code for evaluating state derivatives
        """

        body_lines = self.dy_body(parameters_in_signature, result_name)

        # Add function prototype
        parameters = "" if not parameters_in_signature or \
                     self.oderepr.optimization.parameter_numerals \
                     else "double* parameters, "
        args = "double t, const double* states, {0}double* {1}".format(\
            parameters, result_name)
        dy_function = self.wrap_body_with_function_prototype(\
            body_lines, "rhs", args, \
            "", "Calculate right hand side of {0}".format(self.oderepr.name))
        
        return "\n".join(self.indent_and_split_lines(dy_function))

    def monitored_body(self, parameters_in_signature=False, result_name="monitored"):
        """
        Generate body lines of code for evaluating monitored intermediates
        """

        ode = self.oderepr.ode

        assert(not ode.is_dae)

        # Start building body
        body_lines = ["", "// Assign states"]
        if self.oderepr.optimization.use_state_names:
            for ind, state in enumerate(ode.iter_states()):
                if state.name in self.oderepr._used_in_monitoring["states"]:
                    body_lines.append("const double {0} = states[{1}]".format(\
                        state.name, ind))
        
        # Add parameters code if not numerals
        if parameters_in_signature and \
               not self.oderepr.optimization.parameter_numerals:
            body_lines.append("")
            body_lines.append("// Assign parameters")

            if self.oderepr.optimization.use_parameter_names:
                for ind, param in enumerate(ode.iter_parameters()):
                    if param.name in self.oderepr._used_in_monitoring["parameters"]:
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
                name = "const double " + name
                body_lines.append(self.to_code(expr, name))

        # Add monitored[i] lines
        for ind, (monitored, expr) in enumerate(\
            self.oderepr.iter_monitored_expr()):
            if monitored == "COMMENT":
                body_lines.append("")
                body_lines.append("// " + expr)
            else:
                body_lines.append(self.to_code(expr, "{0}[{1}]".format(\
                    result_name, ind)))

        body_lines.append("")
        
        # Return the body lines
        return body_lines
        
    def monitored_code(self, parameters_in_signature=False, result_name="monitored"):
        """
        Generate code for evaluating monitored intermediates
        """

        body_lines = self.monitored_body(parameters_in_signature, result_name)

        # Add function prototype
        parameters = "" if not parameters_in_signature or \
                     self.oderepr.optimization.parameter_numerals \
                     else "double* parameters, "
        args = "double t, const double* states, {0}double* {1}".format(\
            parameters, result_name)
        monitored_function = self.wrap_body_with_function_prototype(\
            body_lines, "monitored", args, \
            "", "Calculate monitored intermediates {0}".format(self.oderepr.name))
        
        return "\n".join(self.indent_and_split_lines(monitored_function))


class CppCodeGenerator(CCodeGenerator):
    
    def init_language_specific_syntax(self):
        from modelparameters.codegeneration import cppcode
        super(CppCodeGenerator, self).init_language_specific_syntax()
        self.to_code = cppcode

class MatlabCodeGenerator(CodeGenerator):
    """
    A Matlab Code generator
    """
    def init_language_specific_syntax(self):
        from modelparameters.codegeneration import matlabcode

        self.language = "Matlab"
        self.line_ending = ";"
        self.closure_start = ""
        self.closure_end = "end"
        self.line_cont = "..."
        self.comment = "%"
        self.index = lambda i : "({0})".format(i)
        self.indent = 2
        self.indent_str = " "
        self.to_code = matlabcode

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
        
        present_param_comment = ""
        for param in ode.iter_parameters():
            
            if present_param_comment != param.comment:
                present_param_comment = param.comment
                
                body_lines.append("")
                body_lines.append("% --- {0} ---".format(param.comment))
            
            body_lines.append("params.{0} = {1}".format(param.name, param.init))

        body_lines.append("")
            
        # Default initial values and state names
        init_values = [""]
        init_values.append("% --- Default initial state values --- ")
        init_values.append("x0 = zeros({0}, 1)".format(ode.num_states))

        state_names = [""]
        state_names.append("% --- State names --- ")
        state_names.append("state_names = cell({0}, 1)".format(ode.num_states))
        
        present_state_comment = ""
        for ind, state in enumerate(ode.iter_states()):
            
            if present_state_comment != state.comment:
                present_state_comment = state.comment

                init_values.append("")
                init_values.append("% --- {0} ---".format(state.comment))
            
                state_names.append("")
                state_names.append("% --- {0} ---".format(state.comment))

            init_values.append("x0({0}) = {1}".format(ind + 1, state.init))
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

        present_state_comment = ""
        for ind, state in enumerate(ode.iter_states()):
            
            if present_state_comment != state.comment:
                present_state_comment = state.comment
                
                body_lines.append("")
                body_lines.append("% --- {0} ---".format(state.comment))
            
            body_lines.append("{0} = states({1})".format(state.name, ind+1))
        
        # Iterate over any body needed to define the dy
        for expr, name in self.oderepr.iter_dy_body():
            if name == "COMMENT":
                body_lines.append("")
                body_lines.append("% " + expr)
            else:
                body_lines.append(self.to_code(expr, name))

        # Add dy(i) lines
        body_lines.append("dy = zeros({0}, 1)".format(ode.num_states))
        for ind, (state, (derivative, expr)) in enumerate(\
            zip(ode.iter_states(), self.oderepr.iter_derivative_expr())):
            assert(state.sym == derivative[0]), "{0}!={1}".format(state.sym, derivative[0])
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
        
