# flake8: noqa
from gotran import *
from gotran.codegeneration.codegenerator import *

ode = load_ode("winslow")

times = 10000

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

    gen = CodeGenerator(
        ODERepresentation(
            ode,
            keep_intermediates=keep,
            use_cse=use_cse,
            parameter_numerals=numerals,
            use_parameter_names=use_names,
            use_state_names=use_names,
        ),
    )

    # print gen.dy_code()
    code = "\n\n".join(
        [gen.init_states_code(), gen.init_param_code(), gen.dy_code()],
    )  # , gen.monitored_code()])

    params = "{0}_{1}_{2}_{3}".format(
        int(gen.oderepr.optimization.keep_intermediates),
        int(gen.oderepr.optimization.use_cse),
        int(gen.oderepr.optimization.parameter_numerals),
        int(use_names),
    )

    code = "\n".join(
        [
            f"# keep_intermediates = {gen.oderepr.optimization.keep_intermediates}",
            f"# use_cse = {gen.oderepr.optimization.use_cse}",
            f"# parameter_numerals = {gen.oderepr.optimization.parameter_numerals}",
            code,
            "init_states, parameters = init_values(), default_parameters()",
            "import time",
            "t0 = time.time()",
            f"for i in range({times}):",
            "    dy = rhs(init_states, 0.0"
            + (")" if gen.oderepr.optimization.parameter_numerals else ", parameters)"),
            f'''print ""\"
keep_intermediates = {keep}
use_cse            = {use_cse}
parameter_numerals = {numerals}
use_names          = {use_names}""\"''',
            "print 'TIMING: {:.4f}s '.format(time.time()-t0)",
            "import numpy as np",
            """assert(np.sum(np.abs(dy-np.array([ -1.14275465e+00,  -7.71292653e-01,  -1.27524112e-01,   5.52202966e+00,
  -4.89834677e-04,   7.48149270e-05,   1.28800323e-02,  -3.86456175e-03,
  -5.16908177e-05,   3.54021907e-05,   3.11221220e-05,  -2.05612737e-05,
   1.45306827e-06,   1.45247757e-03,   3.14505722e-03,   3.45364437e-04,
  -3.44930000e-04,  -4.34425279e-07,  -1.17425396e-11,   3.77460189e-04,
   5.88955498e-02,  -2.32646624e-06,   5.76675699e-08,   0.00000000e+00,
  -2.39030665e-04,   2.32912903e-04,   1.06656494e-10,   0.00000000e+00,
   0.00000000e+00,   0.00000000e+00])))<1e-6)""",
        ],
    )

    # open("winslow_code_{0}.py".format(params), "w").write(code)

    exec(code)
