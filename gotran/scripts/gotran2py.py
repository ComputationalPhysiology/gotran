#!/usr/bin/env python3

import os

from gotran.model.loadmodel import load_ode
from gotran.codegeneration.codegenerators import PythonCodeGenerator
from gotran.common.options import parameters
from gotran.common import error, info, list_timings, Timer
from modelparameters.parameterdict import *


def gotran2py(filename, params):
    """
    Create a c header file from a gotran model
    """
    timer = Timer("Generate Python code from {}".format(filename))

    # Load Gotran model
    ode = load_ode(filename)

    # Collect monitored
    if params.functions.monitored.generate:
        monitored = [expr.name for expr in ode.intermediates + ode.state_expressions]
    else:
        monitored = None

    # Create a Python code generator
    gen = PythonCodeGenerator(
        params,
        ns=params.namespace,
        import_inside_functions=params.import_inside_functions,
    )

    output = params.output

    if output:
        if not output.endswith(".py"):
            output += ".py"
    else:
        output = filename.replace(".ode", "") + ".py"

    info("")
    info("Generating Python code for the {0} ode...".format(ode.name))
    if params.class_code:
        code = gen.class_code(ode, monitored=monitored)
    else:
        code = gen.module_code(ode, monitored=monitored)
    info("  done.")
    with open(output, "w") as f:
        f.write(code)

    del timer

    if params.list_timings:
        list_timings()


def main():
    import sys, os

    generation_params = parameters.generation.copy()

    params = ParameterDict(
        list_timings=Param(
            False,
            description="If true timings for reading "
            "and evaluating the model is listed.",
        ),
        output=Param("", description="Specify output file name"),
        import_inside_functions=Param(
            False, description="Perform imports inside functions"
        ),
        namespace=OptionParam(
            "math",
            ["math", "np", "numpy", "ufl"],
            description="The math namespace of the generated code",
        ),
        **generation_params
    )
    params.parse_args(usage="usage: %prog FILE [options]")  # sys.argv[2:])

    if len(sys.argv) < 2:
        raise RuntimeError("Expected a single gotran file argument")

    if not os.path.isfile(sys.argv[1]):
        raise IOError("Expected the argument to be a file")

    file_name = sys.argv[1]
    gotran2py(file_name, params)


if __name__ == "__main__":
    main()
