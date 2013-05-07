profile = False

if profile:
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()

from gotran import *

ode = load_ode("tentusscher_panfilov_2006_M_cell.ode")
#ode = load_ode("winslow.ode")

oderepr = ODERepresentation(ode)#, use_cse=True)

model = compile_module(ode, language="Python")

#for expr, name in oderepr.iter_jacobi_body():
#    print name, expr
#
#for indices, expr in oderepr.iter_jacobi_expr():
#    print indices, expr
    
#jacobi, jac_subs, states, cse_subs, cse_jacobi_expr = oderepr.generate_jacobian(True)

if profile:
    pr.disable()
    ps = pstats.Stats(pr)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(0.1)
    print "*"*79
    print "\n"*4
    ps.sort_stats('time')
    ps.print_stats(0.1)
