#!/usr/bin/env python3
from modelparameters.logger import info
from modelparameters.parameterdict import ParameterDict
from modelparameters.parameters import Param
from modelparameters.utils import list_timings

from gotran.codegeneration.codegenerators import CUDACodeGenerator
from gotran.model.loadmodel import load_ode


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
    info(f"Generating CUDA code for the {ode.name} ode...")
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
    import os
    import sys

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
        ),
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
