from gotran.model.expressions2 import *
from gotran.model.loadmodel2 import load_ode
from gotran.codegeneration.algorithmcomponents import *
#from gotran.codegeneration.codegenerator2 import *
from gotran.common.options import parameters
from modelparameters.utils import list_timings, clear_timings
from modelparameters.codegeneration import sympycode
import sys

#parameters["code_generation"]["parameters"]["representation"] = "array"
#parameters["code_generation"]["states"]["representation"] = "array"
parameters["code_generation"]["body"]["representation"] = "reused_array"
#parameters["code_generation"]["body"]["representation"] = "array"
parameters["code_generation"]["body"]["optimize_exprs"] = "numerals_symbols"
parameters["code_generation"]["array"]["flatten"] = True
#parameters["code_generation"]["body"]["use_cse"] = True

ode = load_ode("tentusscher_2004_mcell_updated.ode")
#print "inside script refs", sys.getrefcount(ode)
rhs = rhs_expressions(ode)
#del ode

#list_timings()
#clear_timings()

#pcode = PythonCodeGenerator()
#print "\n".join(pcode._init_states_and_parameters(ode))

#reuse = reuse_body_variables(ode, Intermediate)
comp = componentwise_derivative(ode, 15)
#print "\n".join(pcode._init_states_and_parameters(comp))
jac = jacobian_expressions(ode)
diag = diagonal_jacobian_expressions(jac)
act = jacobian_action_expressions(jac)
act_without = jacobian_action_expressions(jac, with_body=False)
fact = factorized_jacobian_expressions(jac)
lin = linearized_derivatives(ode)

#cse_ode = CommonSubExpressionODE(ode)
#cse_reuse = reuse_body_variables(cse_ode, Intermediate)
#cse_jac = jacobian_expressions(cse_ode)
#cse_jac_reuse = reuse_body_variables(cse_jac, Intermediate, StateExpression)
#cse_act = jacobian_action_expressions(cse_jac)
#cse_fact = factorized_jacobian_expressions(cse_jac)
#cse_lin = linearized_derivatives(cse_ode)
#cse_lin_reuse = reuse_body_variables(cse_lin, Intermediate)
#list_timings()

#for name, expr in iter_body_expressions(\
#    comp_reuse, result_replace=result_replace(ode.state_expressions),
#    state_replace=state_replace(ode),
#    parameter_replace=parameter_replace(ode, numerals=False)):
#    print sympycode(name), sympycode(expr)
#
#for name, expr in iter_body_expressions(\
#    cse_lin_reuse, result_replace=result_replace(cse_ode.state_expressions),
#    state_replace=state_replace(cse_ode),
#    parameter_replace=parameter_replace(cse_ode, numerals=False)):
#    print sympycode(name), sympycode(expr)

#for name, expr in iter_body_expressions(cse_jac_reuse,):
#    result_replace=result_replace(cse_ode.state_expressions)):#,
#    state_replace=state_replace(cse_ode),
#    parameter_replace=parameter_replace(cse_ode, numerals=False)):
#    print sympycode(name), sympycode(expr)

#for name, expr in iter_body_expressions(\
#    ode, result_replace=result_replace(ode.state_expressions)):#,
##    state_replace=state_replace(ode),
##    parameter_replace=parameter_replace(ode, numerals=False))\
#    print sympycode(name), sympycode(expr)
