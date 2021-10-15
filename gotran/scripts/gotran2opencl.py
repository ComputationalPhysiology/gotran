#!/usr/bin/env python3
import os
import sys

from modelparameters.logger import info
from modelparameters.parameterdict import Param
from modelparameters.parameterdict import ParameterDict
from modelparameters.utils import list_timings

from gotran.codegeneration.codegenerators import OpenCLCodeGenerator
from gotran.model.loadmodel import load_ode


def gotran2opencl(filename, params):
    """
    Create a OpenCL file from a gotran model
    """

    # Load Gotran model
    ode = load_ode(filename)

    # Create a C code generator
    gen = OpenCLCodeGenerator(params)

    output = params.output

    if output:
        if not output.endswith(".cl"):
            output += ".cl"
    else:
        output = filename.replace(".ode", "") + ".cl"

    info("")
    info(f"Generating OpenCL code for the {ode.name} ode...")
    code = gen.module_code(ode)
    info("  done.")
    with open(output, "w") as f:
        f.write(code)

    if params.list_timings:
        list_timings()


def main():
    generation_params = OpenCLCodeGenerator.default_parameters()

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
    info(
        "Note: The OpenCL support in gotran is a work in progress. "
        "The CUDA generator is recommended for NVIDIA GPUs.",
    )
    gotran2opencl(file_name, params)


if __name__ == "__main__":
    main()
