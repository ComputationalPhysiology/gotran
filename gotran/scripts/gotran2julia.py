#!/usr/bin/env python3
from modelparameters.logger import info
from modelparameters.parameterdict import ParameterDict
from modelparameters.parameters import Param
from modelparameters.utils import list_timings
from modelparameters.utils import Timer

from gotran.codegeneration.codegenerators import JuliaCodeGenerator
from gotran.model.loadmodel import load_ode


def gotran2julia(filename, params):
    """
    Create a c header file from a gotran model
    """
    timer = Timer(f"Generate Julia code from {filename}")

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
    info(f"Generating Julia code for the {ode.name} ode...")
    code = gen.module_code(ode, monitored=monitored)
    info("  done.")
    with open(output, "w") as f:
        f.write(code)

    del timer

    if params.list_timings:
        list_timings()


def main():
    import os
    import sys

    params = ParameterDict(
        list_timings=Param(
            False,
            description="If true timings for reading "
            "and evaluating the model is listed.",
        ),
        output=Param("", description="Specify output file name"),
        **JuliaCodeGenerator.default_parameters(),
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
