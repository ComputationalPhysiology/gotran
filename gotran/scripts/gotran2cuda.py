#!/usr/bin/env python3

import os

from gotran.model.loadmodel import load_ode
from gotran.codegeneration.codegenerators import CUDACodeGenerator
from gotran.common.options import parameters
from gotran.common import error, list_timings, info
from modelparameters.parameterdict import *


def gotran2cuda(filename, params):
    """
    Create a CUDA file from a gotran model
    """

    # Load Gotran model
    ode = load_ode(filename)

    # Create a C code generator
    gen = CUDACodeGenerator(params)

    output = params.output

    if output:
        if not output.endswith(".cu"):
            output += ".cu"
    else:
        output = filename.replace(".ode", "") + ".cu"

    info("")
    info("Generating CUDA code for the {0} ode...".format(ode.name))
    code = gen.module_code(ode)
    info("  done.")
    with open(output, "w") as f:
        if params.system_headers:
            f.write("#include <math.h>\n")
            f.write("#include <string.h>\n")

        f.write(code)

    if params.list_timings:
        list_timings()


def main():
    import sys, os

    generation_params = CUDACodeGenerator.default_parameters()

    params = ParameterDict(
        list_timings=Param(
            False,
            description="If true timings for reading "
            "and evaluating the model is listed.",
        ),
        system_headers=Param(
            True,
            description="If true system "
            "headers needed to compile moudle is "
            "included.",
        ),
        output=Param("", description="Specify output file name"),
        **dict(
            (name, param)
            for name, param in list(generation_params.items())
            if name not in ["class_code"]
        )
    )
    params.parse_args(usage="usage: %prog FILE [options]")  # sys.argv[2:])

    if len(sys.argv) < 2:
        raise RuntimeError("Expected a single gotran file argument")

    if not os.path.isfile(sys.argv[1]):
        raise IOError("Expected the argument to be a file")

    file_name = sys.argv[1]
    gotran2cuda(file_name, params)


if __name__ == "__main__":
    main()
