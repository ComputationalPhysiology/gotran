#!/usr/bin/env python3

from gotran.model.loadmodel import load_ode
from gotran.codegeneration.codegenerators import MatlabCodeGenerator
from gotran.common.options import parameters
from gotran.common import error, info, list_timings, Timer


def gotran2matlab(filename, params):
    """
    Create a matlab code from a gotran model
    """

    timer = Timer("Generate Matlab code from {}".format(filename))

    # Load Gotran model
    ode = load_ode(filename)

    gen = MatlabCodeGenerator(params)

    info("")
    info("Generating Matlab files for the {0} ode...".format(ode.name))

    # Collect monitored
    if params.functions.monitored.generate:
        monitored = [expr.name for expr in ode.intermediates + ode.state_expressions]
    else:
        monitored = None

    for function_name, code in list(gen.code_dict(ode, monitored).items()):
        open("{0}_{1}.m".format(ode.name, function_name), "w").write(code)

    info("  done.")


def main():
    import os
    import sys

    from modelparameters.parameterdict import ParameterDict, Param

    generation_params = MatlabCodeGenerator.default_parameters()

    params = ParameterDict(
        list_timings=Param(
            False,
            description="If true timings for reading "
            "and evaluating the model is listed.",
        ),
        output=Param("", description="Specify output file name"),
        **generation_params
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
