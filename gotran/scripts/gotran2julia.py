#!/usr/bin/env python3

import os

from gotran.model.loadmodel import load_ode
from gotran.codegeneration.codegenerators import JuliaCodeGenerator
from gotran.common.options import parameters
from gotran.common import error, info, list_timings, Timer
from modelparameters.parameterdict import *


def gotran2julia(filename, params):
    """
    Create a c header file from a gotran model
    """
    timer = Timer("Generate Julia code from {}".format(filename))

    # Load Gotran model
    ode = load_ode(filename)

    # Collect monitored
    if params.functions.monitored.generate:
        monitored = [expr.name for expr in ode.intermediates + ode.state_expressions]
    else:
        monitored = None

    # Create a C code generator
    gen = JuliaCodeGenerator(params)

    output = params.output

    if output:
        if not output.endswith(".jl"):
            output += ".jl"
    else:
        output = filename.replace(".ode", "") + ".jl"

    info("")
    info("Generating Julia code for the {0} ode...".format(ode.name))
    code = gen.module_code(ode, monitored=monitored)
    info("  done.")
    with open(output, "w") as f:
        f.write(code)

    del timer

    if params.list_timings:
        list_timings()


def main():
    import sys, os

    params = ParameterDict(
        list_timings=Param(
            False,
            description="If true timings for reading "
            "and evaluating the model is listed.",
        ),
        output=Param("", description="Specify output file name"),
        **JuliaCodeGenerator.default_parameters()
    )

    params.parse_args(usage="usage: %prog FILE [options]")

    if len(sys.argv) < 2:
        raise RuntimeError("Expected a single gotran file argument")

    if not os.path.isfile(sys.argv[1]):
        raise IOError("Expected the argument to be a file")

    file_name = sys.argv[1]
    gotran2julia(file_name, params)


if __name__ == "__main__":
    main()
