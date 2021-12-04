#!/usr/bin/env python3
from modelparameters.logger import info
from modelparameters.parameterdict import ParameterDict
from modelparameters.parameters import Param
from modelparameters.utils import list_timings
from modelparameters.utils import Timer

from gotran.codegeneration.codegenerators import CppCodeGenerator
from gotran.common.options import parameters
from gotran.model.loadmodel import load_ode


def gotran2cpp(filename, params):
    """
    Create a C++ header file from a gotran model
    """

    timer = Timer(f"Generate C++ code from {filename}")

    # Load Gotran model
    ode = load_ode(filename)

    # Create a C code generator
    gen = CppCodeGenerator(params)

    output = params.output

    if output:
        if not (output.endswith(".cpp") or output.endswith(".h")):
            output += ".h"
    else:
        output = filename.replace(".ode", "") + ".h"

    info("")
    info(f"Generating C++ code for the {ode.name} ode...")
    if params.class_code:
        code = gen.class_code(ode)
    else:
        code = gen.module_code(ode)
    info("  done.")

    with open(output, "w") as f:
        if params.system_headers:
            f.write("#include <cmath>\n")
            f.write("#include <cstring>\n")
            f.write("#include <stdexcept>\n")

        f.write(code)

    del timer

    if params.list_timings:
        list_timings()


def main():
    import os
    import sys

    generation_params = parameters.generation.copy()

    params = ParameterDict(
        list_timings=Param(
            False,
            description="If true timings for reading "
            "and evaluating the model is listed.",
        ),
        output=Param("", description="Specify output file name"),
        system_headers=Param(
            True,
            description="If true system "
            "headers needed to compile moudle is "
            "included.",
        ),
        **generation_params,
    )
    params.parse_args(usage="usage: %prog FILE [options]")  # sys.argv[2:])

    if len(sys.argv) < 2:
        raise RuntimeError("Expected a single gotran file argument")

    if not os.path.isfile(sys.argv[1]):
        raise IOError("Expected the argument to be a file")

    file_name = sys.argv[1]
    gotran2cpp(file_name, params)


if __name__ == "__main__":
    main()
