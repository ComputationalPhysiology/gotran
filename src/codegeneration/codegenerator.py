from collections import deque

from modelparameters.parameters import *

from gotran2.common import check_arg

from oderepresentation import ODERepresentation

class CodeGenerator(object):
    def __init__(self, oderepr):
        check_arg(oderepr, ODERepresentation, 0)
        self.oderepr = oderepr
        self.max_line_length = 79
        self.language = "python"
        self.line_ending = ""
        self.closure_start = ""
        self.closure_end = ""
        self.line_cont = "\\"
        self.comment = "\n#"
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
        
    def dy_code(self):
        """
        Generate code for evaluating state derivatives
        """
        

    def init_code(self):
        """
        Generate code for setting initial condition
        """

        # Start building body
        body_lines = ["# Imports", "import numpy as np",\
                      "from modelparameters.utils import Range", \
                      "", "# Init values"]
        body_lines.append("init_values = np.array([{0}], dtype=np.float_)".format(\
            ", ".join("{0}".format(state.init) for state in \
                      self.oderepr.ode.iter_states())))
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
            body_lines, "init_{0}".format(self.oderepr.name), "**values", \
            "init_values", "Init values")
        
        return "\n".join(self.indent_and_split_lines(init_function))

    def indent_and_split_lines(self, code_lines, indent=0, ret_lines=None):
        """
        Combine a set of lines into a single string
        """

        check_kwarg(indent, "indent", int, ge=0)
        ret_lines = ret_lines or []


        # Walk through the code_lines
        for line in code_lines:

            # If another closure is encountered
            if isinstance(line, list):

                # Add start closure sign if any
                if self.closure_start:
                    ret_lines.append(self.indent*indent*self.indent_str + \
                                     self.closure_start)
                
                ret_lines = self.indent_and_split_lines(line, indent+1, ret_lines)
                
                # Add closure if any
                if self.closure_end:
                    ret_lines.append(self.indent*indent*self.indent_str + \
                                     self.closure_end)

                continue
            if line == "":
                ret_lines.append(line)
                continue

            # Check for long lines
            if self.indent*indent + len(line) + len(self.line_ending) > \
                   self.max_line_length:

                # Divide along white spaces
                splitted_line = deque(line.split(" "))
                first_line = True
                inside_str = False
                while splitted_line:
                    line_stump = []
                    indent_length = self.indent*(indent if first_line else indent + 1)
                    line_length = indent_length
                    first_line = False

                    # FIXME: Line continuation symbol is not included in linelength
                    while splitted_line and ((line_length + len(splitted_line[0]) \
                                              + 1 + inside_str) < self.max_line_length):
                        line_stump.append(splitted_line.popleft())

                        # Add a \" char to first stub if inside str
                        if len(line_stump) == 1 and inside_str:
                            line_stump[-1] = "\""+line_stump[-1]
                            
                        # Check if we get inside or leave a str
                        if "\"" in line_stump[-1] and not \
                               ("\\\"" in line_stump[-1] or "\"\"\"" in line_stump[-1]):
                            inside_str = not inside_str

                        line_length += len(line_stump[-1]) + 1

                    # If we are inside a str and at the end of line add
                    if inside_str:
                        line_stump[-1] = line_stump[-1]+"\""
                        
                    # Join line stump and add indentation
                    ret_lines.append(indent_length*self.indent_str + \
                                     " ".join(line_stump))

                    # If it is the last line stump add line ending otherwise
                    # line continuation sign
                    ret_lines[-1] = ret_lines[-1] + (self.line_cont \
                                    if splitted_line else self.line_ending)
                
            else:
                ret_lines.append("{0}{1}{2}".format(\
                    self.indent*indent*self.indent_str, line, \
                    self.line_ending))

        return ret_lines
