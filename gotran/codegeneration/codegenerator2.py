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
import types
import numpy as np

# Model parameters imports
from modelparameters.parameterdict import *
from modelparameters.codegeneration import ccode, cppcode, pythoncode, \
     sympycode, matlabcode

# Gotran imports
from gotran.common import check_arg, check_kwarg
from gotran.common.options import parameters
from gotran.model.ode2 import ODE
from gotran.model.odeobjects2 import Comment
from gotran.codegeneration.codecomponent import CodeComponent
from gotran.codegeneration.algorithmcomponents import *

__all__ = ["PythonCodeGenerator", "CCodeGenerator", "CppCodeGenerator", \
           "MatlabCodeGenerator"]

def class_name(name):
    check_arg(name, str)
    return name if name[0].isupper() else name[0].upper() + \
           (name[1:] if len(name) > 1 else "")    

def _param_add(params, param_name, default_value, descr, excludes=None, param_type=Param,
               option_check=None):
    """
    Help function to add a parameter
    """
    excludes = excludes or []
    assert(isinstance(params, dict))
    assert(isinstance(param_name, str))
    assert(isinstance(descr, str))
    assert(isinstance(excludes, list))
    assert(issubclass(param_type, Param))
    assert(isinstance(option_check, (types.NoneType, list, str, dict)))

    if param_name in excludes:
        return
    
    kwargs = dict(description=descr)
    args = [default_value]
    
    if option_check is not None:
        if isinstance(option_check, dict):
            kwargs.update(option_check)
        else:
            args.append(option_check)
    
    params[param_name] = param_type(*args, **kwargs)

def _default_code_params(excludes=None):
    excludes = excludes or []
    check_arg(excludes, list, itemtypes=str)

    arg_params = {}

    _param_add(arg_params, "order", "stp", \
               descr="Input argument order: s=states, p=parameters, t=time",\
               param_type=OptionParam, option_check=\
               ["tsp", "stp", "spt", "ts", "st"], \
               excludes=excludes)
    
    _param_add(arg_params, "states_name", "states", \
               descr="Name of the states argument",
               excludes=excludes)
    
    _param_add(arg_params, "parameters_name", "parameters", \
               descr="Name of the parameters argument", \
               excludes=excludes)
    
    _param_add(arg_params, "time_name", "t", \
               descr="Name of the time argument", \
               excludes=excludes)
    
    # Build a dict with allowed parameters
    params = {}

    _param_add(params, "use_state_names", True, \
               "If true, use state names in code (compared to array "\
               "with indices).",\
               excludes=excludes)
    
    _param_add(params, "use_parameter_names", True, \
               "If true, use parameter names in code (compared to array "\
               "with indices).",\
               excludes=excludes)

    _param_add(params, "parameter_numerals", False, \
               "If true, exchange all parameters with their initial numerical "\
               "values.",\
               excludes=excludes)

    _param_add(params, "common_sub_expressions", False, \
               "If true, use sympy common sub expression simplifications.",\
               excludes=excludes)

    _param_add(params, "reuse_intermediates", False, \
               "If true, reuse intermediates to save memory in generated "\
               "code.",\
               excludes=excludes)

    # Add arguments parameters
    params["arguments"] = ParameterDict(**arg_params)
    
    # Return the ParameterDict
    return ParameterDict(**params)

def _default_generation_params(excludes=None):
    exclude = exclude or []
    check_arg(excludes, list, itemtypes=str)

    # Build a dict with allowed parameters
    params = {}

    _param_add(params, "jacobian", False, \
               "If true, generate code for computing the jacobian of the rhs.",\
               excludes)

    _param_add(params, "transposed_jacobian", False, \
               "If true, generate code for computing the transposed jacobian "\
               "of the rhs.",\
               excludes)

    _param_add(params, "jacobian_lu_factorization", False, \
               "If true, generate code for computing the symbolic factorization "\
               "of the jacobian.",\
               excludes)

    _param_add(params, "jacobian_forward_backward_subst", False, \
               "If true, generate code for computing the symbolic forward "\
               "backward substitution of a factoriced jacobian.",\
               excludes)

    _param_add(params, "componentwise_rhs_evaluation", False, \
               "If true, generate code for computing componentwise evaluation "\
               "of the rhs.",\
               excludes)

    _param_add(params, "linearized_rhs_evaluation", False, \
               "If true, generate code for computing linearized evaluation of "\
               "linear rhs terms.",\
               excludes)

    # Return the ParameterDict
    return ParameterDict(**params)

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
    to_code = lambda a,b,c,d : None
    float_types = dict(single="float32", double="float64")

    def __init__(self, params=None):

        params = params or {}

        self.params = self.default_params()
        self.params.update(params)

    @property
    def float_type(self):
        "Return the float type"
        return type(self).float_types[self.params.float_precision]

    @staticmethod
    def default_params():
        # Start out with a copy of the global parameters
        return parameters.code_generation.copy()

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
    to_code = lambda a,b,c,d="math" : pythoncode(b,c,d)
    float_types = dict(single="float32", double="float_")
    
    def args(self, default_arguments=None, result_names=None):
        ret_args=[]
        result_names = result_names or []
        
        default_arguments = default_arguments or self.params.default_arguments
        if self.params.class_code:
            ret_args.append("self")

        for arg in default_arguments:
            if arg == "s":
                ret_args.append(self.params.states.array_name)
            elif arg == "t":
                ret_args.append(self.params.time.name)
            elif arg == "p" and self.params.parameters.representation != \
                     "numerals":
                ret_args.append(self.params.parameters.array_name)
            elif arg == "b" and self.params.body.representation != "named":
                ret_args.append("{0}=None".format(self.params.body.array_name))

        for result_name in result_names:
            ret_args.append("{0}=None".format(result_name))
        
        return ", ".join(ret_args)

    @staticmethod
    def wrap_body_with_function_prototype(body_lines, name, args, comment=""):
        """
        Wrap a passed body of lines with a function prototype
        """
        check_arg(body_lines, list)
        check_arg(name, str)
        check_arg(args, str)
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

        # Append body to prototyp
        prototype.append(body)
        return prototype

    def _init_arguments(self, comp, default_arguments=None):

        check_arg(comp, CodeComponent)

        default_arguments = default_arguments or self.params.default_arguments

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
            
            states_name = self.params.states.array_name
            body_lines.append("")
            body_lines.append("# Assign states")
            body_lines.append("assert(len({0}) == {1})".format(states_name, \
                                                               num_states))
            # Generate state assign code
            if self.params.states.representation == "named":
                
                # If all states are used
                if len(used_states) == len(comp.root.full_states):
                    body_lines.append(", ".join(\
                        state.name for i, state in enumerate(used_states)) + \
                                      " = " + states_name)
                    
                # If only a limited number of states are used
                else:
                    body_lines.append("; ".join(\
                        "{0}={1}[{2}]".format(state.name, states_name, ind) \
                        for ind, state in enumerate(comp.root.full_states) \
                        if state in used_states))
        
        # Add parameters code if not numerals
        if "p" in default_arguments and \
               self.params.parameters.representation in ["named", "array"] and \
               used_parameters:

            parameters_name = self.params.parameters.array_name
            body_lines.append("")
            body_lines.append("# Assign parameters")
            body_lines.append("assert(len({0}) == {1})".format(\
                        parameters_name, num_parameters))
            
            # Generate parameters assign code
            if self.params.parameters.representation == "named":
                
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

        # If using a an array for the body variables
        if self.params.body.representation != "named":
            
            body_name = self.params.body.array_name
            body_lines.append("")
            body_lines.append("# Body array".format(body_name))

            # If passing the body argument to the method
            if  "b" in default_arguments:
                body_lines.append("if {0} is None:".format(body_name))
                body_lines.append(["{0} = np.zeros({1}, dtype=np.{2})".format(\
                    body_name, comp.shapes[body_name]), self.float_type])
                body_lines.append("else:".format(body_name))
                body_lines.append(["assert isinstance({0}, np.ndarray) and "\
                                   "{1}.shape=={2}".format(\
                            body_name, body_name, comp.shapes[body_name])])
            else:
                body_lines.append("{0} = np.zeros({1}, dtype=np.{2})".format(\
                    body_name, comp.shapes[body_name], self.float_type))

        # If initilizing results
        if comp.results:
            body_lines.append("")
            body_lines.append("# Init return args")
            
        for result_name in comp.results:
            shape = comp.shapes[result_name]
            if len(shape) > 1:
                if self.params.array.flatten:
                    shape = (reduce(lambda a,b:a*b, shape, 1),)

            body_lines.append("if {0} is None:".format(result_name))
            body_lines.append(["{0} = np.zeros({1}, dtype=np.{2})".format(\
                result_name, shape, self.float_type)])
            body_lines.append("else:".format(result_name))
            body_lines.append(["assert isinstance({0}, np.ndarray) and "\
            "{1}.shape == {2}".format(result_name, result_name, shape)])
            
        return body_lines

    def function_code(self, comp, indent=0, default_arguments=None, \
                      include_signature=True):

        default_arguments = default_arguments or self.params.default_arguments

        check_arg(comp, CodeComponent)
        check_kwarg(default_arguments, "default_arguments", str)
        check_kwarg(indent, "indent", int)
        
        body_lines = self._init_arguments(comp, default_arguments)
        
        # Iterate over any body needed to define the dy
        for expr in comp.body_expressions:
            if isinstance(expr, Comment):
                body_lines.append("")
                body_lines.append("# " + str(expr))
            else:
                body_lines.append(pythoncode(expr.expr, expr.name))

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
                body_lines, comp.function_name, \
                self.args(default_arguments, comp.results), \
                comp.description)
        
        return "\n".join(self.indent_and_split_lines(body_lines, indent=indent))
        
    def init_states_code(self, ode, indent=0):
        """
        Generate code for setting initial condition
        """

        check_arg(ode, ODE)
        self_arg = self.params.class_code

        # Get all full states
        states = ode.full_states

        # Start building body
        body_lines = ["# Imports", "import numpy as np",\
                      "from modelparameters.utils import Range", \
                      "", "# Init values"]
        body_lines.append("# {0}".format(", ".join("{0}={1}".format(\
            state.name, state.init) for state in states)))
        body_lines.append("init_values = np.array([{0}], dtype=np.{1})"\
                          .format(", ".join("{0}".format(\
                state.init if np.isscalar(state.init) else state.init[0])\
                            for state in states), self.float_type))
        body_lines.append("")
        
        range_check = "lambda value : value {minop} {minvalue} and "\
                      "value {maxop} {maxvalue}"
        body_lines.append("# State indices and limit checker")

        body_lines.append("state_ind = dict({0})".format(\
            ", ".join("{0}=({1}, {2})".format(\
                state.param.name, i, repr(state.param._range))\
                      for i, state in enumerate(states))))
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
        body_lines.append("return init_values")

        args = "self, **values" if self_arg else "**values"
        
        # Add function prototype
        init_function = self.wrap_body_with_function_prototype(\
            body_lines, "init_state_values", args, \
            "Initialize state values")
        
        return "\n".join(self.indent_and_split_lines(init_function, indent=indent))

    def init_parameters_code(self, ode, indent=0):
        """
        Generate code for setting parameters
        """

        check_arg(ode, ODE)
        self_arg = self.params.class_code

        # Get all parameters
        parameters = ode.parameters

        # Start building body
        body_lines = ["# Imports", "import numpy as np",\
                      "from modelparameters.utils import Range", \
                      "", "# Param values"]
        body_lines.append("# {0}".format(", ".join("{0}={1}".format(\
            param.name, param.init) for param in parameters)))
        body_lines.append("init_values = np.array([{0}], dtype=np.{1})"\
                          .format(", ".join("{0}".format(\
                param.init if np.isscalar(param.init) else param.init[0]) \
                    for param in parameters), self.float_type))
        body_lines.append("")
        
        range_check = "lambda value : value {minop} {minvalue} and "\
                      "value {maxop} {maxvalue}"
        body_lines.append("# Parameter indices and limit checker")

        body_lines.append("param_ind = dict({0})".format(\
            ", ".join("{0}=({1}, {2})".format(\
                state.param.name, i, repr(state.param._range))\
                for i, state in enumerate(parameters))))
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
             "init_values[ind] = value"])
            
        body_lines.append("")
        body_lines.append("return init_values")
        
        args = "self, **values" if self_arg else "**values"
        
        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "init_parameter_values", \
            args, "Initialize parameter values")
        
        return "\n".join(self.indent_and_split_lines(function, indent=indent))

    def state_name_to_index_code(self, ode, indent=0):
        """
        Return code for index handling for states
        """
        check_arg(ode, ODE)
        self_arg = self.params.class_code
        states = ode.full_states

        body_lines = []
        body_lines.append("state_inds = dict({0})".format(\
            ", ".join("{0}={1}".format(state.param.name, i) for i, state \
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
        
        args = "self, *states" if self_arg else "*states"
        
        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "state_indices", \
            args, "State indices")
        
        return "\n".join(self.indent_and_split_lines(function, indent=indent))

    def param_name_to_index_code(self, ode, indent=0):
        """
        Return code for index handling for parameters
        """

        check_arg(ode, ODE)
        self_arg = self.params.class_code
        parameters = ode.parameters

        body_lines = []
        body_lines.append("param_inds = dict({0})".format(\
            ", ".join("{0}={1}".format(param.param.name, i) for i, param \
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

        args = "self, *params" if self_arg else "*params"
        
        # Add function prototype
        function = self.wrap_body_with_function_prototype(\
            body_lines, "parameter_indices", \
            args, "Parameter indices")
        
        return "\n".join(self.indent_and_split_lines(function, indent=indent))

    def code_list(self, ode, include_jacobian=True, monitored=None, indent=0):

        monitored = monitored or []
        check_arg(ode, ODE)
        check_kwarg(monitored, "monitored", list, itemtypes=str)
        rhs = rhs_expressions(ode)

        code = [self.init_parameters_code(ode, indent),
                self.init_states_code(ode, indent),
                self.function_code(rhs, indent),
                self.state_name_to_index_code(ode, indent), 
                self.param_name_to_index_code(ode, indent),
                ]

        if monitored:
            code += [self.function_code(monitored_expressions(\
                ode, monitored), indent)]

        if include_jacobian:
            code += [self.function_code(jacobian_expressions(ode), indent)]

        return code

    def class_code(self, ode):
        """
        Generate class code
        """

        check_arg(ode, ODE)
        name = class_name(ode.name)
        code_list = self.code_list(ode, indent=1)

        return  """
class {0}:

    @staticmethod
{1}
""".format(name, "\n\n    @staticmethod\n".join(code_list))

    def module_code(self, ode):

        check_arg(ode, ODE)
        code_list = self.code_list(ode)

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
            prototype.append("// {1}".format(comment))

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
        
