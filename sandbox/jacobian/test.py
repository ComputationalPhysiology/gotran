profile = False

if profile:
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()

from gotran import *

#ode = load_ode("tentusscher_panfilov_2006_M_cell.ode")
ode = load_ode("winslow.ode")

oderepr = ODERepresentation(ode)#, use_cse=True)

oderepr._compute_symbolic_factorization_of_jacobian()

print oderepr._jacobian_factorization_operations
print len(oderepr._jacobian_factorization_operations)

oderepr._compute_symbolic_fb_substitution()

print oderepr._jacobian_fb_substitution_operations
print len(oderepr._jacobian_fb_substitution_operations)

#model = compile_module(ode, language="Python")

#for expr, name in oderepr.iter_jacobi_body():
#    print name, expr
#
#for indices, expr in oderepr.iter_jacobi_expr():
#    print indices, expr
    
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
