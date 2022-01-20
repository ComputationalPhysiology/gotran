import os
import sys
from pathlib import Path

import numpy as np

from gotran.codegeneration.latexcodegenerator import format_expr
from gotran.model.loadmodel import load_ode

_template = """
# {name}

## Parameters
{parameters}

## States
{states}

## State expressions
{state_expr}

## Expressions
{expr}
"""


def make_table(value_matrix, header_names=None):
    max_lengths = np.max([list(map(len, p)) for p in value_matrix], 0)
    template = "|" + "|".join([f"{{{i}:{x}}}" for i, x in enumerate(max_lengths)]) + "|"
    header = ""
    if header_names is not None:
        assert len(header_names) == len(max_lengths)
        header = template.format("Name", "Value")
        header += "\n" + template.format(*["-" * x for x in max_lengths])

    tabular = "\n".join([template.format(*p) for p in value_matrix])
    return "\n".join([header, tabular])


def gotran2md(filename):

    filename = Path(filename)
    ode = load_ode(filename)

    parameters = make_table(
        [[f"${format_expr(p.name)}$", f"{p._repr_latex_()}"] for p in ode.parameters],
        header_names=["Name", "Value"],
    )
    states = make_table(
        [[f"${format_expr(p.name)}$", f"{p._repr_latex_()}"] for p in ode.states],
        header_names=["Name", "Value"],
    )

    expr = "\n\n".join(
        [
            f"$$\n{p._repr_latex_name()} = {p._repr_latex_expr()}\n$$"
            for p in ode.intermediates
        ],
    )

    state_expr = "\n\n".join(
        [
            f"$$\n{p._repr_latex_name()} = {p._repr_latex_expr()}\n$$"
            for p in ode.state_expressions
        ],
    )

    mdname = filename.with_suffix(".md")
    with open(mdname, "w") as f:

        f.write(
            _template.format(
                name=ode.name,
                parameters=parameters,
                states=states,
                expr=expr,
                state_expr=state_expr,
            ),
        )

    print(f"Saved to {mdname}")


def main():
    if len(sys.argv) < 2:
        raise RuntimeError("Expected a single gotran file argument")

    if not os.path.isfile(sys.argv[1]):
        raise IOError("Expected the argument to be a file")

    file_name = sys.argv[1]
    gotran2md(file_name)
