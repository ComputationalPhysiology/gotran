#!/usr/bin/env python3

from gotran.codegeneration.codegenerators import MatlabCodeGenerator
from gotran.common import Timer, error, info, list_timings
from gotran.common.options import parameters
from gotran.model.loadmodel import load_ode


def gotran2matlab(filename, params):
    """
    Create a matlab code from a gotran model
    """

    timer = Timer(f"Generate Matlab code from {filename}")

    # Load Gotran model
    ode = load_ode(filename)

    gen = MatlabCodeGenerator(params)

    info("")
    info(f"Generating Matlab files for the {ode.name} ode...")

    # Collect monitored
    if params.functions.monitored.generate:
        monitored = [expr.name for expr in ode.intermediates + ode.state_expressions]
    else:
        monitored = None

    for function_name, code in list(gen.code_dict(ode, monitored).items()):
        open(f"{ode.name}_{function_name}.m", "w").write(code)

    info("  done.")


def main():
    import os
    import sys

    from modelparameters.parameterdict import Param, ParameterDict

    generation_params = MatlabCodeGenerator.default_parameters()

    params = ParameterDict(
        list_timings=Param(
            False,
            description="If true timings for reading "
            "and evaluating the model is listed.",
        ),
        output=Param("", description="Specify output file name"),
        **generation_params,
    )
    params.parse_args(usage="usage: %prog FILE [options]")  # sys.argv[2:])

    if len(sys.argv) < 2:
        raise RuntimeError("Expected a single gotran file argument")

    if not os.path.isfile(sys.argv[1]):
        raise IOError("Expected the argument to be a file")

    file_name = sys.argv[1]
    gotran2matlab(file_name, params)


if __name__ == "__main__":
    main()
