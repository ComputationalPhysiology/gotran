from gotran import *

ode = load_ode("tentusscher_2004_mcell")

oderepr = ODERepresentation(ode, use_cse=True)
#oderepr._compute_linearized_dy_cse()

for expr, name in oderepr.iter_linerized_body():
    print name, expr

for idx, expr in oderepr.iter_linerized_expr():
    print idx, expr

for idx, (subs, expr) in enumerate(oderepr.iter_componentwise_dy()):
    print
    print idx
    for sub in subs:
        print sub[1], sub[0]
    print expr
