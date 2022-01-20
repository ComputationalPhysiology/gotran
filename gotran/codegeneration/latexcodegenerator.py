# Copyright (C) 2014 George Bahij / Johan Hake
#
# This file is part of Gotran.
#
# Gotran is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Gotran is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Gotran. If not, see <http://www.gnu.org/licenses/>.
import re
import string
import tokenize
from io import StringIO

from modelparameters import sympy
from modelparameters.codegeneration import latex as mp_latex
from modelparameters.parameterdict import ParameterDict
from modelparameters.parameters import Param

from ..model.expressions import Expression
from ..model.expressions import StateDerivative

__all__ = ["LatexCodeGenerator"]

# Latex templates
_global_opts = """\\setkeys{{breqn}}{{breakdepth={{1}}}}
{GLOBALOPTS}"""

_latex_template = """\\documentclass[a4paper,{FONTSIZE}pt]{{article}}
{PKGS}
{PREOPTS}

\\begin{{document}}
{OPTS}
{BODY}
{ENDOPTS}

\\end{{document}}"""

_param_table_template = """
% ---------------- BEGIN PARAMETERS ---------------- %

\\{SECTIONTYPE}*{{Parameters}}\n
\\label{{sec:ODE_Parameters}}
{OPTS}
\\begin{{longtabu}}{{| l l {PDESCCELLSTYLE} |}}
\\caption[Parameter Table]{{\\textbf{{Parameter Table}}}}\\\\
\\hline
\\multicolumn{{1}}{{|c}}{{\\textbf{{Parameter\\hspace{{0.5cm}}}}}} &
\\multicolumn{{1}}{{c}}{{\\textbf{{Value\\hspace{{0.5cm}}}}}} &
\\multicolumn{{1}}{{c|}}{{\\textbf{{Description\\hspace{{0.5cm}}}}}}\\\\ \\hline
\\endfirsthead
\\multicolumn{{3}}{{c}}%
{{{{\\bfseries\\tablename\\ \\thetable{{}} --- continued from previous page}}}}
\\\\ \\hline
\\multicolumn{{1}}{{|c}}{{\\textbf{{Parameter\\hspace{{0.5cm}}}}}} &
\\multicolumn{{1}}{{c}}{{\\textbf{{Value\\hspace{{0.5cm}}}}}} &
\\multicolumn{{1}}{{c|}}{{\\textbf{{Description\\hspace{{0.5cm}}}}}}\\\\ \\hline
\\endhead
\\hline
\\multicolumn{{3}}{{|r|}}{{{{Continued on next page}}}}\\\\ \\hline
\\endfoot
\\hline \\hline
\\endlastfoot
{BODY}\\\\
\\hline
\\end{{longtabu}}
{ENDOPTS}

% ----------------- END PARAMETERS ----------------- %
"""

_state_table_template = """
% ---------------- BEGIN STATES ---------------- %

\\{SECTIONTYPE}*{{Initial Values}}\n
\\label{{sec:ODE_States}}
{OPTS}
\\begin{{longtabu}}{{| l l {SDESCCELLSTYLE} |}}
\\caption[State Table]{{\\textbf{{State Table}}}}\\\\
\\hline
\\multicolumn{{1}}{{|c}}{{\\textbf{{State\\hspace{{0.5cm}}}}}} &
\\multicolumn{{1}}{{c}}{{\\textbf{{Value\\hspace{{0.5cm}}}}}} &
\\multicolumn{{1}}{{c|}}{{\\textbf{{Description\\hspace{{0.5cm}}}}}}\\\\ \\hline
\\endfirsthead
\\multicolumn{{3}}{{c}}%
{{{{\\bfseries\\tablename\\ \\thetable{{}} --- continued from previous page}}}}
\\\\ \\hline
\\multicolumn{{1}}{{|c}}{{\\textbf{{State\\hspace{{0.5cm}}}}}} &
\\multicolumn{{1}}{{c}}{{\\textbf{{Value\\hspace{{0.5cm}}}}}} &
\\multicolumn{{1}}{{c|}}{{\\textbf{{Description\\hspace{{0.5cm}}}}}}\\\\ \\hline
\\endhead
\\hline
\\multicolumn{{3}}{{|r|}}{{{{Continued on next page}}}}\\\\ \\hline
\\endfoot
\\hline \\hline
\\endlastfoot
{BODY}\\\\
\\hline
\\end{{longtabu}}
{ENDOPTS}

% ----------------- END STATES ----------------- %
"""

_components_template = """
% ---------------- BEGIN COMPONENTS ---------------- %

\\{SECTIONTYPE}*{{Components}}
\\label{{sec:ODE_Components}}
{OPTS}
{BODY}
{ENDOPTS}

% ----------------- END COMPONENTS ----------------- %
"""

_greek = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lamda "
    "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega".split(" ")
)


def format_expr(expr, mul_symbol="dot", **print_settings):
    if isinstance(expr, str) and expr in _greek:
        return f"\\{expr}"
    # Some values are treated as special cases by sympy.sympify.
    # Return these as they are.
    if isinstance(expr, str) and expr in [x for x in dir(sympy) if len(x) == 1]:
        return expr
    return mp_latex(sympy.sympify(expr), mul_symbol=mul_symbol, **print_settings)


def _default_latex_params():
    """
    Initializes default parameters.
    """

    params = dict()

    # Specify output file
    params["output"] = Param("", description="Specify LaTeX output file")

    # Set number of columns per page
    # FIXME: ScalarParam might cause parse_args() to crash due to non-ASCII
    # symbols.
    # params["page_columns"] = ScalarParam(
    #     1, ge=1, description="Set number of columns per page in "
    #     "LaTeX document")
    params["page_columns"] = Param(
        1,
        description="Set number of columns per page in " "LaTeX document",
    )

    # Set equation font size
    # FIXME: ScalarParam might cause parse_args() to crash due to non-ASCII
    # symbols.
    # params["font_size"] = ScalarParam(
    #     10, ge=1, description="Set global font size for LaTeX document")
    params["font_size"] = Param(
        10.0,
        description="Set global font size for LaTeX document",
    )

    # Set font size for mathematical expressions.
    # FIXME: ScalarParam might cause parse_args() to crash due to non-ASCII
    # symbols.
    # params["math_font_size"] = ScalarParam(
    #    1, ge=1, description="Set font size for mathematical "
    #    "expressions in LaTeX document. Uses global font size if left "
    #    "blank")
    params["math_font_size"] = Param(
        0.0,
        description="Set font size for mathematical expressions in "
        "LaTeX document. Uses global font size if left blank",
    )

    # Toggle bold equation labels
    params["bold_equation_labels"] = Param(
        True,
        description="Give equation labels a bold typeface in " "LaTeX document",
    )

    # If set to False, does not generate the preamble
    params["preamble"] = Param(
        True,
        description="If set to False, LaTeX document will be "
        "be generated without the preamble",
    )

    # If set to true, sets document to a landscape page layout
    params["landscape"] = Param(
        False,
        description="Set LaTeX document to landscape layout",
    )

    # Latex separator between factors in products
    params["mul_symbol"] = Param(
        "dot",
        description="Multiplication symbol for Sympy LatexPrinter",
    )

    # Flag to enable page numbers
    params["page_numbers"] = Param(True, description="Enable page numbers")

    # Flag to disable state table (currently unused)
    # params["no_state_descriptions"] = Param(
    #     False, description="Disable table column for state descriptions")

    # Flag to disable parameter table (currently unused)
    # params["no_parameter_descriptions"] = Param(
    #     False, description="Disable table column for parameter descriptions")

    # Set headline types for States, Parameters and Components
    params["section_type"] = Param(
        "section",
        description="Section type (e.g. 'section', 'subsection')",
    )

    # Set page margins
    params["margins"] = Param(
        "",
        description="Set page margins (e.g. '0.75in'). Uses LaTeX "
        "defaults if left blank",
    )

    # Set column seperator distance
    params["columnsep"] = Param(
        "",
        description="Set column separator distance (e.g. '0.25cm'). "
        "Uses LaTeX default if left blank",
    )

    # Set column separator line width
    params["columnseprule"] = Param(
        "",
        description="Set column separator line width (e.g. '0.2pt'). "
        "Uses LaTeX default if left blank",
    )

    # Flag to let the code generator attempt automatically converting
    # state and parameter names in descriptions to math-mode
    params["auto_format_description"] = Param(
        False,
        description="Automatically format state and parameter " "descriptions",
    )

    # Flag to toggle numbering style for equations.
    params["equation_subnumbering"] = Param(
        True,
        description="Use component-wise equation subnumbering",
    )

    params["parameter_description_cell_style"] = Param(
        "l",
        description="Set description cell type for the parameter table. "
        "Use 'X' for long descriptions, or 'p{5cm}' to set a fixed 5 cm",
    )

    params["state_description_cell_style"] = Param(
        "l",
        description="Set description cell type for the state table. "
        "Use 'X' for long descriptions, or 'p{5cm}' to set a fixed 5 cm",
    )

    # Return the ParameterDict
    return ParameterDict(**params)


class LatexCodeGenerator(object):
    packages = [
        ("fullpage", ""),
        ("longtable,tabu", ""),
        ("multicol", ""),
        ("amsmath", ""),
        ("mathpazo", ""),
        ("flexisym", "[mathpazo]"),
        ("breqn", ""),
    ]
    print_settings = dict()

    def __init__(self, ode, params=None):
        self.ode = ode
        self._name = ode.name
        self.params = params if params else _default_latex_params()

        self.output_file = params.output or f"{ode.name}.tex"

        # Verify valid multiplication symbol
        if self.params.mul_symbol in ("dot", "ldot", "times"):
            self.print_settings["mul_symbol"] = self.params.mul_symbol
        else:
            self.print_settings["mul_symbol"] = None

        if self.params.margins:
            self.packages.append(("geometry", "[margin=" + self.params.margins + "]"))

    def generate(self, params=None):
        """
        Generate a LaTeX-formatted document describing a Gotran ODE.
        """
        params = params if params else self.params
        latex_output = ""

        global_opts = self.format_global_options(_global_opts)

        if not params.preamble:
            latex_output = (
                self.generate_parameter_table()
                + self.generate_state_table()
                + self.generate_components()
            )
        else:
            document_opts = self.format_options(
                override=["font_size", "landscape", "page_numbers"],
            )
            latex_output = _latex_template.format(
                FONTSIZE=params.font_size,
                PKGS=self.format_packages(self.packages),
                PREOPTS=global_opts,
                OPTS=document_opts["begin"],
                BODY=self.generate_parameter_table()
                + self.generate_state_table()
                + self.generate_components(),
                ENDOPTS=document_opts["end"],
            )

        return latex_output

    def generate_parameter_table(self, params=None):
        """
        Return a LaTeX-formatted string for a longtable describing the ODE's
        parameters.
        """
        params = params if params else self.params
        param_str = "\\\\\n".join(
            self.format_param_table_row(par) for par in self.ode.parameters
        )
        param_table_opts = self.format_options(exclude=["page_columns"])
        param_table_output = _param_table_template.format(
            SECTIONTYPE=params["section_type"],
            PDESCCELLSTYLE=params["parameter_description_cell_style"],
            OPTS=param_table_opts["begin"],
            BODY=param_str,
            ENDOPTS=param_table_opts["end"],
        )
        return param_table_output

    def generate_state_table(self, params=None):
        """
        Return a LaTeX-formatted string for a longtable describing the ode's
        states.
        """
        params = params if params else self.params
        state_str = "\\\\\n".join(
            self.format_state_table_row(state) for state in self.ode.full_states
        )
        state_table_opts = self.format_options(exclude=["page_columns"])
        state_table_output = _state_table_template.format(
            SECTIONTYPE=params["section_type"],
            SDESCCELLSTYLE=params["state_description_cell_style"],
            OPTS=state_table_opts["begin"],
            BODY=state_str,
            ENDOPTS=state_table_opts["end"],
        )
        return state_table_output

    def generate_components(self, params=None):
        """
        Return a LaTeX-formatted string of the ODE's derivative components and
        intermediate calculations.
        """
        params = params if params else self.params
        components_str = ""
        comp_template = (
            "{LABEL}\n\\label{{comp:{LABELID}}}\n"
            "\\begin{{dgroup{SUBNUM}}}\n"
            "{BODY}\\end{{dgroup{SUBNUM}}}\n"
        )
        eqn_template = (
            "  \\begin{{dmath}}\n    \\label{{eq:{0}}}\n"
            "    {1} = {2}\\\\\n  \\end{{dmath}}\n"
        )

        subnumbering = "" if params["equation_subnumbering"] else "*"

        for comp in self.ode.components:
            if comp.rates:
                body = [
                    obj
                    for obj in comp.ode_objects
                    if isinstance(obj, Expression)
                    and not isinstance(obj, StateDerivative)
                ]
            else:
                body = [obj for obj in comp.ode_objects if isinstance(obj, Expression)]

            if not body:
                continue

            format_label = self.format_component_label(comp.name)
            label_id = comp.name.replace(" ", "_")
            format_body = ""

            # Iterate over all objects of the component
            for obj in body:
                format_body += eqn_template.format(
                    obj.name,
                    obj._repr_latex_name(),
                    obj._repr_latex_expr(),
                )

            components_str += comp_template.format(
                LABEL=format_label,
                LABELID=label_id,
                BODY=format_body,
                SUBNUM=subnumbering,
            )

        components_opts = self.format_options(
            override=["page_columns", "math_font_size"],
        )
        components_output = _components_template.format(
            SECTIONTYPE=params["section_type"],
            OPTS=components_opts["begin"],
            BODY=components_str,
            ENDOPTS=components_opts["end"],
        )
        return components_output

    def format_param_table_row(self, param):
        """
        Return a LaTeX-formatted string for a longtable row describing a
        parameter.
        E.g.:
        >>> LatexCodeGenerator.format_param_table_row(
        ...     Parameter("g_earth",
        ...         ScalarParam(9.81, unit="m/s**2",
        ...                     description="Surface gravity"))
        '  $g_{earth}$\\hspace{0.5cm} & $9.81 \\mathrm{\\frac{m}{s^{2}}}$
        \\hspace{0.5cm} & Surface gravity'
        """
        return (
            "  ${NAME}$\\hspace{{0.5cm}} & {VAL}"
            "\\hspace{{0.5cm}} & {DESC}".format(
                NAME=self.format_expr(param.name),
                VAL=param._repr_latex_(),
                DESC=self.format_description(param.param.description, param.name),
            )
        )

    def format_state_table_row(self, state):
        """
        Return a LaTeX-formatted string for a longtable row describing a
        state's initial value.
        E.g.:
        >>> LatexCodeGenerator.format_state_table_row("amu",
        ... "Atomic mass unit", 931.46, "MeV/c**2")
        '  $amu$\\hspace{0.5cm} & $931.46 \\mathrm{\\frac{MeV}{c^{2}}}$
        \\hspace{0.5cm} & Atomic mass unit'
        """
        return (
            "  ${NAME}$\\hspace{{0.5cm}} & {VAL}"
            "\\hspace{{0.5cm}} & {DESC}".format(
                NAME=self.format_expr(state.name),
                VAL=state._repr_latex_(),
                DESC=self.format_description(state.param.description, state.name),
            )
        )

    def format_component_label(self, label):
        """
        Return a LaTeX-formatted string of an ODE component group label.
        """
        label_opts = self.format_options(override=["bold_equation_labels"])
        return "{0}{1}{2}\\\\".format(
            label_opts["begin"],
            label.replace("_", "\\_"),
            label_opts["end"],
        )

    def format_description(self, description, name):
        """
        If auto-format flag is set, attempt to automatically format description,
        setting references to states and parameters in math mode.
        """

        if not self.params["auto_format_description"]:
            return description + "\\hspace{0.5cm}"

        formatted_description = ""
        first = True
        for ttype, token, _, _, _ in tokenize.generate_tokens(
            StringIO(description).readline,
        ):
            if tokenize.ISEOF(ttype):
                break

            formatted_token = token
            mathmode = False

            if any([c in token for c in ("_", "^")]) or token == name:
                # or token in [state.name for state in self.ode.full_states] \
                # or token in [par.name for par in self.ode.parameters]:
                mathmode = True

            if mathmode:
                for c in ("_", "^"):
                    i = formatted_token.find(c)
                    while i != -1:
                        formatted_token = (
                            formatted_token[: i + 1]
                            + "{"
                            + formatted_token[i + 1 :]
                            + "}"
                        )
                        i = formatted_token.find(c, i + 1)
                formatted_token = "$" + formatted_token + "$"

            if token not in string.punctuation and not first:
                formatted_token = " " + formatted_token

            formatted_description += formatted_token
            first = False

        return formatted_description + "\\hspace{0.5cm}"

    def format_options(self, exclude=None, override=None, params=None):
        """
        Wrap options around a LaTeX string, excluding specified elements.
        If override is not empty, only those elements will be used, ignoring
        exclude.
        """
        exclude = exclude if exclude and not override else []
        override = override or []
        opts = _default_latex_params()
        opts.update(params if params else self.params)

        begin_str = end_str = ""

        if opts.page_columns > 1 and (
            ("page_columns" not in exclude and not override)
            or "page_columns" in override
        ):
            begin_str = f"\\begin{{multicols}}{{{opts.page_columns}}}\n" + begin_str
            end_str += "\\end{multicols}\n"

        # Non-standard options -- only include if specified in override:

        if "font_size" in override:
            begin_str = (
                "{{\\fontsize{{{0}}}{{{1:.1f}}}\\selectfont\n".format(
                    opts.font_size,
                    opts.font_size * 1.2,
                )
                + begin_str
            )
            end_str += "}% end fontsize\n"

        if "math_font_size" in override and opts.math_font_size:
            begin_str = (
                "{{\\fontsize{{{0}}}{{{1:.1f}}}\n".format(
                    opts.math_font_size,
                    opts.math_font_size * 1.2,
                )
                + begin_str
            )
            end_str += "}% end fontsize\n"

        if "bold_equation_labels" in override and opts.bold_equation_labels:
            begin_str = "\\textbf{" + begin_str
            end_str += "}"

        if "landscape" in override and opts.landscape:
            if "pdflscape" not in (package_name for package_name, _ in self.packages):
                self.packages.append(("pdflscape", ""))
            begin_str = "\\begin{landscape}\n" + begin_str
            end_str += "\\end{landscape}\n"

        if "page_numbers" in override and not opts.page_numbers:
            begin_str = "\\pagenumbering{gobble}\n" + begin_str

        return {"begin": begin_str, "end": end_str}

    def format_global_options(self, option_template, params=None):
        """
        Inject additional options into the global option template
        """
        opts = _default_latex_params()
        opts.update(params if params else self.params)

        additional_options = list()

        if opts.columnsep:
            additional_options.append(f"\\setlength{{\\columnsep}}{{{opts.columnsep}}}")

        if opts.columnseprule:
            additional_options.append(
                f"\\setlength{{\\columnseprule}}{{{opts.columnseprule}}}",
            )

        global_options = option_template.format(
            GLOBALOPTS="\n".join(additional_options),
        )

        return global_options

    def format_expr(self, expr):
        """
        Return a LaTeX-formatted string for a sympy expression.
        E.g.:
        >>> LatexCodeGenerator.format_expr("exp(i*pi) + 1")
        'e^{i \\pi} + 1'
        """
        return format_expr(expr, **self.print_settings)

    def format_unit(self, unit):
        """
        Return sympified and LaTeX-formatted string describing given unit.
        E.g.:
        >>> LatexCodeGenerator.format_unit("m/s**2")
        '\\mathrm{\\frac{m}{s^{2}}}'
        """
        atomic_units = re.findall(r"([a-zA-Z]+)", unit)
        atomic_dict = dict((au, sympy.Symbol(au)) for au in atomic_units)
        sympified_unit = eval(unit, atomic_dict, dict())
        return "\\mathrm{{{0}}}".format(mp_latex(sympified_unit), **self.print_settings)

    def format_packages(self, package_list):
        """
        Return list of packages and options as a LaTeX-formatted string.
        Assumes package list is on the form (("package1", "[options1]"), ...).
        """
        return "\n".join(
            f"\\usepackage{option}{{{package}}}" for package, option in package_list
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.ode)}, {repr(self.params)})"
