from gotran.model.expressions2 import *
from gotran.model.loadmodel import load_ode
from gotran.codegeneration.algorithmcomponents import *
from gotran.codegeneration.codegenerator import *
from gotran.common.options import parameters
from gotran.common import set_log_level, DEBUG

from modelparameters.utils import list_timings, clear_timings
from modelparameters.codegeneration import sympycode
import sys

#set_log_level(DEBUG)

ode = load_ode("tentusscher_2004_mcell_updated.ode")

code_params = parameters["code_generation"].copy()

#monitored = monitored_expressions(ode, ["i_NaK", "i_NaCa", "i_CaL", "d_fCa"])

#parameters["code_generation"]["parameters"]["representation"] = "array"
#parameters["code_generation"]["states"]["representation"] = "array"
#parameters["code_generation"]["body"]["representation"] = "reused_array"
#parameters["code_generation"]["parameters"]["representation"] = "named"
#parameters["code_generation"]["states"]["representation"] = "named"
#parameters["code_generation"]["body"]["representation"] = "reused_array"
#parameters["code_generation"]["body"]["representation"] = "named"
#parameters["code_generation"]["body"]["optimize_exprs"] = "numerals_symbols"
#parameters["code_generation"]["array"]["flatten"] = True
#parameters["code_generation"]["body"]["use_cse"] = True
#parameters["code_generation"]["body"]["use_cse"] = True
#parameters["code_generation"]["float_precision"] = "single"

#ode = load_ode("tentusscher_2004_mcell_updated.ode")

#monitored = monitored_expressions(ode, ["i_NaK", "i_NaCa", "i_CaL", "d_fCa"])

code_params["body"]["optimize_exprs"] = "none"
code_params["body"]["representation"] = "array"
code_params["body"]["array_name"] =  "JADA"
code_params["parameters"]["representation"] = "named"
code_params["states"]["representation"] = "array"
code_params["states"]["array_name"] = "STATES"
code_params["body"]["use_cse"] = False
code_params["float_precision"] = "double"

from goss.codegeneration2 import GossCodeGenerator
from goss.compilemodule import jit

#gcg = GossCodeGenerator(ode, ["V"], ["g_Na", "g_Kr"], ["E_Na", "E_Ca"])
#print gcg.class_code()

ode = jit(ode, ["V"], ["g_Na", "g_Kr"], ["E_Na", "E_Ca"])


#codegen = PythonCodeGenerator(code_params)
#jac = jacobian_expressions(ode, params=code_params)
#rhs = rhs_expressions(ode, params=code_params)
#monitored =monitored_expressions(ode, ["i_NaK", "i_NaCa", "i_CaL", "d_fCa"], \
#                                 params=code_params)

#print codegen.function_code(rhs)

#rhs = rhs_expressions(ode)
#comp = componentwise_derivative(ode, 15)
#jac = jacobian_expressions(ode)
#lin = linearized_derivatives(ode)

#diag = diagonal_jacobian_expressions(jac)
#act = jacobian_action_expressions(jac)
#act_without = jacobian_action_expressions(jac, with_body=False)
#fact = factorized_jacobian_expressions(jac)

#codegen = CCodeGenerator()

#print CCodeGenerator().module_code(ode)
#print
#print "#################################"
#print CppCodeGenerator().class_code(ode)

#print codegen.init_states_code(ode)
#print codegen.init_parameters_code(ode)
#print codegen.state_name_to_index_code(ode)
#print codegen.param_name_to_index_code(ode)

#module_code = codegen.module_code(ode)
#print module_code
#exec(module_code)

#states = init_state_values()
#parameters = init_parameter_values()
#
#import numpy as np
#import math
#
#print rhs(states, 0.0, parameters)
#print compute_jacobian(states, 0.0, parameters)

#print codegen.function_code(rhs)
#print codegen.function_code(monitored)
#print codegen.function_code(comp, include_signature=False)
#print codegen.function_code(jac)
#print codegen.function_code(lin)
#print codegen.componentwise_body(ode)


#cse_rhs = rhs_expressions(ode)
#cse_comp = componentwise_derivative(ode, 15)
#cse_jac = jacobian_expressions(ode)
#cse_diag = diagonal_jacobian_expressions(jac)
#cse_act = jacobian_action_expressions(jac)
#cse_act_without = jacobian_action_expressions(jac, with_body=False)
#cse_fact = factorized_jacobian_expressions(jac)
#cse_lin = linearized_derivatives(ode)

#list_timings()
