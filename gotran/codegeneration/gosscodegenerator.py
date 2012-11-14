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

# FIXME: Move to goss python layer

# Gotran imports
from codegenerator import CppCodeGenerator

_class_template = """#ifndef {MODELNAME}_H_IS_INCLUDED
#define {MODELNAME}_H_IS_INCLUDED

#include <boost/shared_ptr.hpp>
#include <stdexcept>
#include <cmath>

#include \"goss/ParameterizedODE.h\"
#include \"goss/LinearizedODE.h\"

namespace goss {{

  // Implementation of gotran generated ODE
  class {ModelName} : public ParameterizedODE {linearized_inheritance}
  {{
  public:

    // Constructor
    {ModelName}() : ODE({num_states}),
      ParameterizedODE({num_states}, {num_parameters}, {num_field_states}, {num_field_parameters}, {num_monitored}), {linearized_base_initialization}
{variable_initialization}
      
    {{
{constructor}
    }}

    // Evaluate rhs of the ODE
    void eval(const double* states, double t, double* values)
    {{
{eval_code}
    }}
{eval_componentwise_code}
    // Get default initial conditions
    void get_ic(goss::DoubleVector *values) const
    {{
{initial_condition_code}
    }}

    // Return a copy of the ODE
    boost::shared_ptr<ODE> copy() const
    {{
      return new {ModelName}(*this);
    }}

    // Evaluate the monitored intermediates
    void eval_monitored(const double* states, double t, double* monitored) const
    {{
{monitored_evaluation_code}
    }}

    // Set all field parameters
    void set_field_parameters(const double* field_params)
    {{
{set_field_parameters_code}
    }}

  private:
{variable_declaration}    

  }};

}}

#endif
"""

_no_monitored_snippet = """\n      // No monitored
      throw std::runtime_error(\"No monitored in the \\'{0}\\' model.\");"""

_class_form = dict(
  MODELNAME="NOT_IMPLEMENTED",
  ModelName="NOT_IMPLEMENTED",
  linearized_inheritance="",
  linearized_base_initialization="",
  num_states="NOT_IMPLEMENTED",
  num_parameters=0,
  num_field_states=0,
  num_field_parameters=0,
  num_monitored=0,
  state_names_ctr="NOT_IMPLEMENTED",
  variable_initialization="NOT_IMPLEMENTED",
  constructor="",
  eval_code="NOT_IMPLEMENTED",
  initial_condition_code="NOT_IMPLEMENTED",
  eval_componentwise_code="",
  monitored_evaluation_code="",
  set_field_parameters_code="",
  variable_declaration="NOT_IMPLEMENTED",
)

class GossCodeGenerator(CppCodeGenerator):
    """
    Class for generating an implementation of a goss ODE
    """
    
    def __init__(self, oderepr):
        
        # Init base class
        super(GossCodeGenerator, self).__init__(oderepr)
        
        self.class_form = _class_form.copy()
        name = self.oderepr.name

        self._name = name if name[0].isupper() else name[0].upper() + \
                     (name[1:] if len(name) > 1 else "")
        
        self.class_form["MODELNAME"] = name.upper()
        self.class_form["ModelName"] = self.name
            
        self.class_form["num_states"] = self.oderepr.ode.num_states
        self.class_form["num_parameters"] = self.oderepr.ode.num_parameters
        self.class_form["num_field_states"] = self.oderepr.ode.num_field_states
        self.class_form["num_field_parameters"] = \
                            self.oderepr.ode.num_field_parameters
        self.class_form["num_monitored"] = \
                            self.oderepr.ode.num_monitored_intermediates

        self.class_form["monitored_evaluation_code"] = \
                _no_monitored_snippet.format(oderepr.name.capitalize()) + \
                "\n"

        self._constructor_body()
        self._variable_init_and_declarations()
        self._eval_code()
        if oderepr.ode.num_monitored_intermediates > 0:
            self._monitored_code()

    @property
    def name(self):
        return self._name
    
    def generate(self):
        """
        Generate the goss ode code
        """
        return _class_template.format(**self.class_form)

    def _constructor_body(self):
        """
        Generate code snippets for constructor
        """

        ode = self.oderepr.ode

        # State names
        state_names = [state.name for state in ode.iter_states()]
        field_state_names = [state.name for state in ode.iter_field_states()]
        body = ["", "// State names"]
        body.extend("_state_names[{0}] = \"{1}\"".format(i, name) \
                    for i, name in enumerate(state_names))

        # Parameter names
        if self.class_form["num_parameters"] > 0:
            body.extend(["", "// Parameter names"])
            body.extend("_parameter_names[{0}] = \"{1}\"".format(\
                i, param.name) for i, param in \
                        enumerate(ode.iter_parameters()))
            
        # Field state names
        if self.class_form["num_field_states"] > 0:
            body.extend(["", "// Field state names"])
            body.extend("_field_state_names[{0}] = \"{1}\"".format(\
                i, name) for i, name in enumerate(field_state_names))

            body.extend(["", "// Field state indices"])
            for i, name in enumerate(field_state_names):
                body.append("_field_state_indices[{0}] = {1}".format(\
                    i, state_names.index(name)))
            
        # Field parameter names
        if self.class_form["num_field_parameters"] > 0:
            body.extend(["", "// Field parameter names"])
            body.extend("_field_parameter_names[{0}] = \"{1}\"".format(\
                i, param.name) for i, param in \
                        enumerate(ode.iter_field_parameters()))
            
        # Monitored names
        if self.class_form["num_monitored"] > 0:
            body.extend(["", "// Monitored names"])
            body.extend("_monitored_names[{0}] = \"{1}\"".format(\
                i, monitored) for i, (monitored, expr) in \
                        enumerate(ode.iter_monitored_intermediates()))

        # Parameter to value map
        if self.class_form["num_parameters"] > 0:
            body.extend(["", "// Parameter to value map"])
            body.extend("_param_to_value[\"{0}\"] = &{1}".format(\
                param.name, param.name) for i, param in \
                        enumerate(ode.iter_parameters()))

        body.append("")
        code = "\n".join(self.indent_and_split_lines(body, indent=3))
            
        self.class_form["constructor"] = code


    def _variable_init_and_declarations(self):
        """
        Generate code snippets for variable declarations, initialization and
        initial conditions
        """

        ode = self.oderepr.ode

        state_declarations = ["", "// State assignments"]
        parameter_declarations = []
        init = []

        state_declarations.extend("const double {0} = states[{1}]".format(state.name, i) \
                                  for i, state in enumerate(ode.iter_states()))
        
        # Parameter declaration and init
        if self.class_form["num_parameters"] > 0:
            parameter_declarations.extend(["", "// Parameters"])
            parameter_declarations.append("double " + ", ".join(\
                param.name for param in ode.iter_parameters())) 
            
            init.extend("{0}({1})".format(param.name, param.init[0] \
                        if param.is_field else param.init) \
                        for param in ode.iter_parameters())
        
        # Parameter initialization
        init = [", ".join(init)]
        code = "\n".join(self.indent_and_split_lines(init, indent=3, \
                                                     no_line_ending=True))
        self.class_form["variable_initialization"] = code
        
        # State declarations
        code = "\n".join(self.indent_and_split_lines(state_declarations,\
                                                     indent=3))
        self.class_form["state_declarations"] = code
        
        # Parameter declaration
        code = "\n".join(self.indent_and_split_lines(parameter_declarations,\
                                                     indent=2))
        self.class_form["variable_declaration"] = code

        # Initial condition code
        ic_code = ["", "// Initial conditions"]
        ic_code.append("values->n = _num_states")
        ic_code.append("values->data.reset(new double[_num_states])")
        ic_code.extend("values->data[{0}] = {1}".format(\
            i, state.init[0] if state.is_field else state.init) \
                       for i, state in enumerate(ode.iter_states()))

        code = "\n".join(self.indent_and_split_lines(ic_code, indent=3))

        self.class_form["initial_condition_code"] = code

        # Field parameter setting
        set_field_parameters_code = []
        
        if self.class_form["num_field_parameters"] > 0:
            set_field_parameters_code.extend(["", "// Set field parameters"])
            set_field_parameters_code.extend("{0} = field_params[{1}]".format(\
                param.name, i) for i, param in \
                                    enumerate(ode.iter_field_parameters()))

            set_field_parameters_code.append("")
            
            code = "\n".join(self.indent_and_split_lines(\
                set_field_parameters_code, indent=3))
            self.class_form["set_field_parameters_code"] = code
        
        

    def _eval_code(self):
        """
        Generate code for the eval method(s)
        """
        
        body = self.dy_body(parameters_in_signature=False, \
                            result_name="values")
        code = "\n".join(self.indent_and_split_lines(body, indent=3))
        self.class_form["eval_code"] = code
        
    def _monitored_code(self):
        """
        Generate code for the monitored method
        """
        body = self.monitored_body(parameters_in_signature=False, \
                                   result_name="monitored")
        code = "\n".join(self.indent_and_split_lines(body, indent=3))
        self.class_form["monitored_evaluation_code"] = code
        
        
