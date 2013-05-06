from gotran import *

ode = load_ode("tentusscher_panfilov_2006_M_cell.ode")
#ode = load_ode("winslow.ode")


oderepr = ODERepresentation(ode)
pcode = CodeGenerator(oderepr)

print pcode.jacobian_code()

#for indices, expr in oderepr.iter_jacobian_expr():
#    print indices, expr
    
#jacobi, jac_subs, states, cse_subs, cse_jacobi_expr = oderepr.generate_jacobian(True)
