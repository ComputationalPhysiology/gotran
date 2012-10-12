__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2012 " + __author__
__date__ = "2012-08-22 -- 2012-10-09"
__license__  = "GNU LGPL Version 3.0 or later"

# System imports
from collections import deque

# Model parameters imports
from modelparameters.parameters import *

# Gotran imports
from gotran2.common import check_arg
from oderepresentation import ODERepresentation

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
        args = "t, states"
        if not self.oderepr.optimization.parameter_numerals:
            args += ", parameters"
        
        dy_function = self.wrap_body_with_function_prototype(\
            body_lines, "dy_{0}".format(self.oderepr.name), args, \
            "dy", "Calculate right hand side")
        
        return "\n".join(self.indent_and_split_lines(dy_function))

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
                          .format(", ".join("{0}".format(state.init) \
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
             ["raise ValueError(\"{{0}} is not a state in the {0} ODE\"."\
              "format(state_name))".format(self.oderepr.name)],
             "ind, range = state_ind[state_name]",
             "if value not in range:",
             ["raise ValueError(\"While setting \'{0}\' {1}\".format("\
              "state_name, range.format_not_in(value)))"],
             "", "# Assign value",
             "init_values[ind] = value"])
            
        body_lines.append("")
        
        # Add function prototype
        init_function = self.wrap_body_with_function_prototype(\
            body_lines, "{0}_init_values".format(self.oderepr.name), \
            "**values", "init_values", "Init values")
        
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
             ["raise ValueError(\"{{0}} is not a param in the {0} ODE\"."\
              "format(param_name))".format(self.oderepr.name)],
             "ind, range = param_ind[param_name]",
             "if value not in range:",
             ["raise ValueError(\"While setting \'{0}\' {1}\".format("\
              "param_name, range.format_not_in(value)))"],
             "", "# Assign value",
             "param_values[ind] = value"])
            
        body_lines.append("")
        
        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "{0}_parameters".format(self.oderepr.name), \
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
                             "\"\"\"" in line_stump[-1])):
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
            body_lines, "dy_{0}".format(self.oderepr.name), args, \
            "", "Calculate right hand side of {0}".format(self.oderepr.name))
        
        return "\n".join(self.indent_and_split_lines(dy_function))


class CppCodeGenerator(CCodeGenerator):
    
    def init_language_specific_syntax(self):
        from modelparameters.codegeneration import cppcode
        super(CppCodeGenerator, self).init_language_specific_syntax()
        self.to_code = cppcode
