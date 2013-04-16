# Copyright (C) 2013 George Bahij / Johan Hake
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

# System imports
import re

# Model parameters imports
from modelparameters.codegeneration import latex as mp_latex
from modelparameters.parameters import Param, ScalarParam
from modelparameters.parameterdict import ParameterDict

# Other imports
import sympy

# Latex templates
_global_opts = "\\setkeys{breqn}{breakdepth={1}}"

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

\\section*{{Parameters}}\n
{OPTS}
\\begin{{longtable}}{{| l l p{{4cm}} |}}
\\caption[Parameter Table]{{\\textbf{{Parameter Table}}}}\\\\
\\hline
\\multicolumn{{1}}{{|c}}{{\\textbf{{Parameter\\hspace{{0.5cm}}}}}} &
\\multicolumn{{1}}{{c}}{{\\textbf{{Value\\hspace{{0.5cm}}}}}} &
\\multicolumn{{1}}{{c|}}{{\\textbf{{Description}}}}\\\\ \\hline
\\endfirsthead
\\multicolumn{{3}}{{c}}%
{{{{\\bfseries\\tablename\\ \\thetable{{}} --- continued from previous page}}}}
\\\\ \hline
\\multicolumn{{1}}{{|c}}{{\\textbf{{Parameter\\hspace{{0.5cm}}}}}} &
\\multicolumn{{1}}{{c}}{{\\textbf{{Value\\hspace{{0.5cm}}}}}} &
\\multicolumn{{1}}{{c|}}{{\\textbf{{Description}}}}\\\\ \\hline
\\endhead
\\hline
\\multicolumn{{3}}{{|r|}}{{{{Continued on next page}}}}\\\\ \\hline
\\endfoot
\\hline \\hline
\\endlastfoot
{BODY}\\\\
\\hline
\\end{{longtable}}
{ENDOPTS}

% ----------------- END PARAMETERS ----------------- %
"""

_state_table_template = """
% ---------------- BEGIN STATES ---------------- %

\\section*{{Initial Values}}\n
{OPTS}
\\begin{{longtable}}{{| l l |}}
\\caption[State Table]{{\\textbf{{State Table}}}}\\\\
\\hline
\\multicolumn{{1}}{{|c}}{{\\textbf{{State\\hspace{{0.5cm}}}}}} &
\\multicolumn{{1}}{{c|}}{{\\textbf{{Value}}}}\\\\ \\hline
\\endfirsthead
\\multicolumn{{2}}{{c}}%
{{{{\\bfseries\\tablename\\ \\thetable{{}} --- continued from previous page}}}}
\\\\ \hline
\\multicolumn{{1}}{{|c}}{{\\textbf{{State\\hspace{{0.5cm}}}}}} &
\\multicolumn{{1}}{{c|}}{{\\textbf{{Value}}}}\\\\ \\hline
\\endhead
\\hline
\\multicolumn{{2}}{{|r|}}{{{{Continued on next page}}}}\\\\ \\hline
\\endfoot
\\hline \\hline
\\endlastfoot
{BODY}\\\\
\\hline
\\end{{longtable}}
{ENDOPTS}

% ----------------- END STATES ----------------- %
"""

_components_template = """
% ---------------- BEGIN COMPONENTS ---------------- %

\\section*{{Components}}
{OPTS}
{BODY}
{ENDOPTS}

% ----------------- END COMPONENTS ----------------- %
"""


def _default_latex_params():
    """
    Initializes default parameters.
    """

    params = dict()

    # Specify output file
    params["latex_output"] = Param(
        "", description="Specify LaTeX output file")

    # Set number of columns per page
    # FIXME: ScalarParam might cause parse_args() to crash due to non-ASCII
    # symbols.
    # params["page_columns"] = ScalarParam(
    #     1, ge=1, description="Set number of columns per page in "
    #     "LaTeX document")
    params["page_columns"] = Param(
        1, description="Set number of columns per page in "
        "LaTeX document")

    # Set equation font size
    # FIXME: ScalarParam might cause parse_args() to crash due to non-ASCII
    # symbols.
    # params["font_size"] = ScalarParam(
    #     10, ge=1, description="Set global font size for LaTeX document")
    params["font_size"] = Param(
        10, description="Set global font size for LaTeX document")

    # Set font size for mathematical expressions.
    # FIXME: ScalarParam might cause parse_args() to crash due to non-ASCII
    # symbols.
    #params["math_font_size"] = ScalarParam(
    #    1, ge=1, description="Set font size for mathematical "
    #    "expressions in LaTeX document. Uses global font size if left "
    #    "blank")
    params["math_font_size"] = Param(
        0, description="Set font size for mathematical expressions in "
        "LaTeX document. Uses global font size if left blank")

    # Toggle bold equation labels
    params["bold_equation_labels"] = Param(
        True, description="Give equation labels a bold typeface in "
        "LaTeX document")

    # If set to True, does not generate the preamble
    params["no_preamble"] = Param(
        False, description="If set to True, LaTeX document will be "
        "be generated without the preamble")

    # If set to true, sets document to a landscape page layout
    params["landscape"] = Param(
        False, description="Set LaTeX document to landscape layout")

    # Latex separator between factors in products
    params["mul_symbol"] = Param(
        "dot", description="Multiplication symbol for Sympy LatexPrinter")

    # Return the ParameterDict
    return ParameterDict(**params)


class LatexCodeGenerator(object):
    packages = [("fullpage", ""), ("longtable", ""), ("multicol", ""),
                ("amsmath", ""), ("mathpazo", ""), ("flexisym", "[mathpazo]"),
                ("breqn", "")]
    print_settings = dict()

    def __init__(self, ode, params=None):
        self.ode = ode
        self._name = ode.name
        self.params = params if params else _default_latex_params()

        self.output_file = params.latex_output or '{0}.tex'.format(
            ode.name)

        # Verify valid multiplication symbol
        if self.params.mul_symbol in ("dot", "ldot", "times"):
            self.print_settings["mul_symbol"] = self.params.mul_symbol
        else:
            self.print_settings["mul_symbol"] = None

    def generate(self, params=None):
        """
        Generate a LaTeX-formatted document describing a Gotran ODE.
        """
        params = params if params else self.params
        latex_output = ""

        if params.no_preamble:
            latex_output = self.generate_parameter_table() \
                          + self.generate_state_table() \
                          + self.generate_components()
        else:
            document_opts = self.format_options(
                override=["font_size", "landscape"])
            latex_output = _latex_template.format(
                FONTSIZE=params.font_size,
                PKGS=self.format_packages(self.packages),
                PREOPTS=_global_opts, OPTS=document_opts["begin"],
                BODY=self.generate_parameter_table()
                     + self.generate_state_table()
                     + self.generate_components(),
                ENDOPTS=document_opts["end"])

        return latex_output

    def generate_parameter_table(self):
        """
        Return a LaTeX-formatted string for a longtable describing the ODE's
        parameters.
        """
        param_str = "\\\\\n".join(
            self.format_param_table_row(par.name,
                                        par.param.description,
                                        par.value,
                                        par.param.unit)
            for par in self.ode.parameters)
        param_table_opts = self.format_options(exclude=["page_columns"])
        param_table_output = _param_table_template.format(
            OPTS=param_table_opts["begin"], BODY=param_str,
            ENDOPTS=param_table_opts["end"])
        return param_table_output

    def generate_state_table(self):
        """
        Return a LaTeX-formatted string for a longtable describing the ode's
        states.
        """
        state_str = "\\\\\n".join(
            self.format_state_table_row(state.name,
                                        state.value,
                                        state.param.unit)
            for state in self.ode.states)
        state_table_opts = self.format_options(exclude=["page_columns"])
        state_table_output = _state_table_template.format(
            OPTS=state_table_opts["begin"], BODY=state_str,
            ENDOPTS=state_table_opts["end"])
        return state_table_output

    def generate_components(self):
        """
        Return a LaTeX-formatted string of the ODE's derivative components and
        intermediate calculations.
        """
        components_str = ""
        comp_template = "{LABEL}\n\\begin{{dgroup*}}\n{BODY}\\end{{dgroup*}}\n"
        eqn_template = \
            "  \\begin{{dmath}}\n    {0} = {1}\\\\\n  \\end{{dmath}}\n"

        for name, comp in self.ode.components.items():
            if not (comp.intermediates or comp.derivatives):
                continue
            format_label = self.format_component_label(name)
            format_body = ""
            for intermediate in comp.intermediates:
                format_body += eqn_template.format(
                    self.format_expr(intermediate.name),
                    self.format_expr(intermediate.expr))
            for derivative in comp.derivatives:
                if len(derivative.states) == 1:
                    format_body += eqn_template.format(
                        "\\frac{{d{0}}}{{dt}}".format(
                            self.format_expr(derivative.states[0].name)),
                        self.format_expr(derivative.expr))
                else:
                    # TODO: Take a closer look at this...
                    format_body += eqn_template.format(
                        self.format_expr(derivative.expr),
                        self.format_expr(derivative.expr))
            components_str += comp_template.format(LABEL=format_label,
                                                   BODY=format_body)

        components_opts = \
            self.format_options(override=["page_columns", "math_font_size"])
        components_output = _components_template.format(
            OPTS=components_opts["begin"], BODY=components_str,
            ENDOPTS=components_opts["end"])
        return components_output

    def format_param_table_row(self, param_name, description, value, unit="1"):
        """
        Return a LaTeX-formatted string for a longtable row describing a
        parameter.
        E.g.:
        >>> LatexCodeGenerator.format_param_table_row("g_earth",
        ... "Surface gravity", 9.81, "m/s**2")
        '  $g_{earth}$\\hspace{0.5cm} & $9.81 \\mathrm{\\frac{m}{s^{2}}}$
        \\hspace{0.5cm} & Surface gravity'
        """
        return "  ${NAME}$\\hspace{{0.5cm}} & ${VAL}{UNIT}$" \
               "\\hspace{{0.5cm}} & {DESC}".format(
                   NAME=self.format_expr(param_name), VAL=value,
                   DESC=description,
                   UNIT=" " + self.format_unit(unit) if unit != "1" else "")

    def format_state_table_row(self, state_name, value, unit="1"):
        """
        Return a LaTeX-formatted string for a longtable row describing a
        state's initial value.
        E.g.:
        >>> LatexCodeGenerator.format_state_table_row("amu", 931.46,
        ... "MeV/c**2")
        '  $amu$\\hspace{0.5cm} & $931.46 \\mathrm{\\frac{MeV}{c^{2}}}$'
        """
        return "  ${NAME}$\\hspace{{0.5cm}} & ${VAL}{UNIT}$".format(
            NAME=self.format_expr(state_name), VAL=value,
            UNIT=" " + self.format_unit(unit) if unit != "1" else "")

    def format_component_label(self, label):
        """
        Return a LaTeX-formatted string of an ODE component group label.
        """
        label_opts = self.format_options(override=["bold_equation_labels"])
        return "{0}{1}{2}\\\\".format(label_opts["begin"],
                                      label.replace("_", "\\_"),
                                      label_opts["end"])

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

        if opts.page_columns > 1 \
            and (("page_columns" not in exclude and not override)
                 or "page_columns" in override):
            begin_str = "\\begin{{multicols}}{{{0}}}\n".format(
                opts.page_columns) + begin_str
            end_str += "\\end{multicols}\n"

        # Non-standard options -- only include if specified in override:

        if "font_size" in override:
            begin_str = "{{\\fontsize{{{0}}}{{{1:.1f}}}\n".format(
                opts.font_size, opts.font_size*1.2) + begin_str
            end_str += "}% end fontsize\n"

        if "math_font_size" in override and opts.math_font_size:
            begin_str = "{{\\fontsize{{{0}}}{{{1:.1f}}}\n".format(
                opts.math_font_size, opts.math_font_size*1.2) + begin_str
            end_str += "}% end fontsize\n"

        if "bold_equation_labels" in override and opts.bold_equation_labels:
            begin_str = "\\textbf{" + begin_str
            end_str += "}"

        if "landscape" in override and opts.landscape:
            if not "pdflscape" in \
                    (package_name for package_name, _ in self.packages):
                self.packages.append(("pdflscape", ""))
            begin_str = "\\begin{landscape}\n" + begin_str
            end_str += "\\end{landscape}\n"

        return {"begin": begin_str, "end": end_str}

    def format_expr(self, expr):
        """
        Return a LaTeX-formatted string for a sympy expression.
        E.g.:
        >>> LatexCodeGenerator.format_expr("exp(i*pi) + 1")
        'e^{i \\pi} + 1'
        """
        # Some values are treated as special cases by sympy.sympify.
        # Return these as they are.
        if isinstance(expr, str) and expr in ["beta", "gamma"]:
            return "\\{0}".format(expr)
        if isinstance(expr, str) and expr in ["I", "O"]:
            return expr
        return mp_latex(sympy.sympify(expr), **self.print_settings)

    def format_unit(self, unit):
        """
        Return sympified and LaTeX-formatted string describing given unit.
        E.g.:
        >>> LatexCodeGenerator.format_unit("m/s**2")
        '\\mathrm{\\frac{m}{s^{2}}}'
        """
        atomic_units = re.findall(r"([a-zA-Z]+)", unit)
        atomic_dict = dict((au, sympy.Symbol(au)) for au in atomic_units)
        sympified_unit = eval(unit, atomic_dict, {})
        return "\\mathrm{{{0}}}".format(mp_latex(sympified_unit),
                                        **self.print_settings)

    def format_packages(self, package_list):
        """
        Return list of packages and options as a LaTeX-formatted string.
        Assumes package list is on the form (("package1", "[options1]"), ...).
        """
        return "\n".join("\\usepackage{0}{{{1}}}".format(option, package)
                         for package, option in package_list)

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, repr(self.ode), repr(self.params))