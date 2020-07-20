#!/usr/bin/env python3

from gotran.codegeneration.latexcodegenerator import (
    LatexCodeGenerator,
    _default_latex_params,
)
from gotran.model.loadmodel import load_ode
from gotran import info


def gotran2latex(filename, params):
    """
    Create LaTeX code from a gotran model
    """

    if not params.sympy_contraction:
        # A hack to avoid sympy contractions
        import gotran.codegeneration.avoidsympycontractions

    # Load Gotran model
    ode = load_ode(filename)

    # Check for completeness
    # TODO: Should raise a more descriptive exception?
    if not ode.is_complete:
        raise Exception("Incomplete ODE")

    # Create a gotran -> LaTeX document generator
    gen = LatexCodeGenerator(ode, params)

    info("")
    info("Generating LaTeX files for the {0} ode...".format(ode.name))
    with open(gen.output_file, "w") as f:
        f.write(gen.generate())
    info("  done.")


def main():
    import os
    import sys
    from modelparameters.parameterdict import ParameterDict, Param

    params = _default_latex_params()
    params = ParameterDict(
        sympy_contraction=Param(
            True,
            description="If True sympy contraction"
            " will be used, turning (V-3)/2 into V/2-3/2",
        ),
        **params
    )
    params.parse_args(usage="usage: %prog FILE [options]")

    if len(sys.argv) < 2:
        raise RuntimeError("Expected a single gotran file argument")

    if not os.path.isfile(sys.argv[1]):
        raise IOError("Expected the argument to be a file")

    filename = sys.argv[1]
    gotran2latex(filename, params)


if __name__ == "__main__":
    main()
