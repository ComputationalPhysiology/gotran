#!/usr/bin/env python3
import os
import sys

from modelparameters.parameterdict import ParameterDict
from modelparameters.parameters import Param

from gotran.codegeneration.algorithmcomponents import rhs_expressions
from gotran.codegeneration.codegenerators import DOLFINCodeGenerator
from gotran.model.loadmodel import load_ode


def gotran2dolfin(filename, params):
    """
    Create ufl code from a gotran model
    """

    # Load Gotran model
    ode = load_ode(filename)

    # Create a ufl based model code generator
    code_gen = DOLFINCodeGenerator(params)

    output = params.output

    if output:
        if not output.endswith(".py"):
            output += ".py"
    else:
        output = filename.replace(".ode", "") + ".py"

    f = open(output, "w")
    f.write(code_gen.init_states_code(ode) + "\n\n")
    f.write(code_gen.init_parameters_code(ode) + "\n\n")
    f.write(
        code_gen.function_code(rhs_expressions(ode, params=code_gen.params.code))
        + "\n",
    )


def main():

    params = ParameterDict(
        output=Param("", description="Specify output file name"),
        **DOLFINCodeGenerator.default_parameters()
    )
    params.parse_args(usage="usage: %prog FILE [options]")  # sys.argv[2:])

    if len(sys.argv) < 2:
        raise RuntimeError("Expected a single gotran file argument")

    if not os.path.isfile(sys.argv[1]):
        raise IOError("Expected the argument to be a file")

    file_name = sys.argv[1]
    gotran2dolfin(file_name, params)


if __name__ == "__main__":
    main()
