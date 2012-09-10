from gotran2 import *
from gotran2.codegeneration.codegenerator import *

ode = load_ode("winslow")

times = 10000

for keep, use_cse, numerals, use_names in \
        [(1,0,0,1), (1,0,0,0), \
         (1,0,1,1), (1,0,1,0), \
         (0,0,0,1), (0,0,0,0), \
         (0,0,1,1), (0,0,1,0), \
         (0,1,0,1), (0,1,0,0), \
         (0,1,1,1), (0,1,1,0)]:
    
    gen = CodeGenerator(ODERepresentation(ode, keep_intermediates=keep, \
                                          use_cse=use_cse, parameter_numerals=numerals,\
                                          use_names=use_names))
    
    #print gen.dy_code()
    code ="\n\n".join([gen.init_states_code(),
                       gen.init_param_code(),
                       gen.dy_code()])
    
    params = "{0}_{1}_{2}_{3}".format(\
        int(gen.oderepr.optimization.keep_intermediates),
        int(gen.oderepr.optimization.use_cse),           
        int(gen.oderepr.optimization.parameter_numerals),
        int(gen.oderepr.optimization.use_names))
    
    code = "\n".join(\
        ["# keep_intermediates = {0}".format(gen.oderepr.optimization.keep_intermediates),
         "# use_cse = {0}".           format(gen.oderepr.optimization.use_cse),
         "# parameter_numerals = {0}".format(gen.oderepr.optimization.parameter_numerals),
         code,
         "init_states, parameters = winslow_init_values(), winslow_parameters()",
         "import time", "t0 = time.time()",
         "for i in range({0}):".format(times),
         "    dy = dy_winslow(0.0, init_states" + (")" \
         if gen.oderepr.optimization.parameter_numerals else ", parameters)"),
         '''print """
keep_intermediates = {0}
use_cse            = {1}
parameter_numerals = {2}
use_names          = {3}"""'''.format(keep, use_cse, numerals, use_names),
         "print 'TIMING: {:.4f}s '.format(time.time()-t0)",
         "import numpy as np",
         """assert(np.sum(np.abs(dy-np.array([ -1.16545396e+00,  -7.71292653e-01,  -1.27524112e-01,
         5.52202966e+00,  -4.89834677e-04,   7.48149270e-05,
         1.28800323e-02,  -3.86456175e-03,  -4.88979642e-05,
         3.54021907e-05,   3.11221220e-05,  -2.05612737e-05,
         1.45306827e-06,   3.45364437e-04,  -3.44930000e-04,
        -4.34425279e-07,  -1.17425396e-11,  -5.88888669e-02,
         5.88948983e-02,   8.65013548e-08,   0.00000000e+00,
         0.00000000e+00,  -2.39030665e-04,   2.32912829e-04,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   3.77460189e-04,   1.45247757e-03,
         3.14505722e-03])))<1e-6)"""
         ])

    #open("winslow_code_{0}.py".format(params), "w").write(code)
    
    exec(code)

