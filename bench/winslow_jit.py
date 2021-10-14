from gotran import *
from gotran.codegeneration.codegenerator import *
from gotran.codegeneration.compilemodule import compile_module
import time
import numpy as np

ode = load_ode("winslow")

# oderepr = ODERepresentation(ode, keep_intermediates=False, \
#                            use_cse=True)#, parameter_numerals=True)

times = 100000
keep, use_cse, numerals, use_names = (1, 0, 0, 1)

oderepr = ODERepresentation(
    ode,
    keep_intermediates=keep,
    use_cse=use_cse,
    parameter_numerals=numerals,
    use_names=use_names,
)
gen = CodeGenerator(oderepr)

exec (gen.init_states_code())
exec (gen.init_param_code())
exec (gen.dy_code())

parameters = default_parameters()
states = init_values()
dy = np.asarray(states).copy()
dy_correct = rhs(states, 0.0, parameters)

for keep, use_cse, numerals, use_names in [
    (1, 0, 0, 1),
    (1, 0, 0, 0),
    (1, 0, 1, 1),
    (1, 0, 1, 0),
    (0, 0, 0, 1),
    (0, 0, 0, 0),
    (0, 0, 1, 1),
    (0, 0, 1, 0),
    (0, 1, 0, 1),
    (0, 1, 0, 0),
    (0, 1, 1, 1),
    (0, 1, 1, 0),
]:

    oderepr = ODERepresentation(
        ode,
        keep_intermediates=keep,
        use_cse=use_cse,
        parameter_numerals=numerals,
        use_parameter_names=use_names,
        use_state_names=use_names,
    )

    params = "{0}_{1}_{2}_{3}".format(
        int(oderepr.optimization.keep_intermediates),
        int(oderepr.optimization.use_cse),
        int(oderepr.optimization.parameter_numerals),
        int(use_names),
    )

    module = compile_module(oderepr)
    t0 = time.time()
    if oderepr.optimization.parameter_numerals:
        module.rhs(states, 0.0, dy)
        for i in range(times):
            module.rhs(states, 0.0, dy)
    else:
        module.rhs(states, 0.0, parameters, dy)
        for i in range(times):
            module.rhs(states, 0.0, parameters, dy)

    print """
keep_intermediates = {0}
use_cse            = {1}
parameter_numerals = {2}
use_names          = {3}""".format(
        keep, use_cse, numerals, use_names
    )
    print "TIMING: {:.4f}s".format(time.time() - t0)

    assert np.sum(np.abs(dy - dy_correct)) < 1e-6
