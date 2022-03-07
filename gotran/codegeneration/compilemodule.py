# Copyright (C) 2012 Johan Hake
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
import hashlib
import sys
import types
import typing
from enum import Enum

from modelparameters.logger import debug
from modelparameters.logger import info
from modelparameters.logger import value_error
from modelparameters.utils import check_arg
from modelparameters.utils import check_kwarg

from .. import __version__
from ..common.options import parameters
from ..model.loadmodel import load_ode
from ..model.ode import ODE
from .codegenerators import class_name
from .codegenerators import CppCodeGenerator
from .codegenerators import DOLFINCodeGenerator
from .codegenerators import PythonCodeGenerator

try:
    import cppyy

    _has_cppyy = True
    cppyy_version = cppyy.__version__
except ImportError:
    _has_cppyy = False
    cppyy_version = 0


def has_cppyy() -> bool:
    return _has_cppyy


module_template = """import dijitso as _dijitso
import numpy as _np
from cppyy.gbl import {clsname}
_module = {clsname}()

{code}
"""


rhs_template = """def {rhs_function_name}({args},{rhs_name} = None):
    '''
    Evaluates the right hand side of the model

    Arguments
    ---------
{args_doc}
    {rhs_name} : np.ndarray (optional)
        The computed result
    '''
    import numpy as np
    if {rhs_name} is None:
        {rhs_name} = np.zeros_like({states_name})

    _module.{rhs_function_name}({args}, {rhs_name})
    return {rhs_name}
"""


jacobian_template = """def {jacobian_function_name}({args}, {jac_name}=None):
    '''
    Evaluates the jacobian of the model

    Arguments
    ---------
{args_doc}
    {jac_name} : np.ndarray (optional)
        The computed result
    '''
    import numpy as np
    if {jac_name} is None:
        {jac_name} = np.zeros({num_states}*{num_states}, dtype=np.float_)
    elif not isinstance({jac_name}, np.ndarray):
        raise TypeError(\"expected a NumPy array.\")
    elif len({jac_name}.shape) != 2 or {jac_name}.shape[0] != {jac_name}.shape[1] or {jac_name}.shape[0] != {num_states}:
        raise ValueError(\"expected a square shaped matrix with size ({num_states}, {num_states})\")
    else:
        # Flatten Matrix
        {jac_name}.shape = ({num_states}*{num_states},)

    _module.{jacobian_function_name}({args}, {jac_name})
    {jac_name}.shape = ({num_states},{num_states})
    return {jac_name}
"""

monitor_template = """def {monitored_function_name}({args}, {monitored_name}=None):
    '''
    Evaluates any monitored intermediates of the model

    Arguments
    ---------
{args_doc}
    {monitored_name} : np.ndarray (optional)
        The computed result
    '''
    import numpy as np
    if {monitored_name} is None:
        {monitored_name} = np.zeros({num_monitored}, dtype=np.float_)
    elif not isinstance({monitored_name}, np.ndarray):
        raise TypeError(\"expected a NumPy array.\")
    elif len({monitored_name}) != {num_monitored}:
        raise ValueError(\"expected a numpy array of size: {num_monitored}\")

    _module.{monitored_function_name}({args}, {monitored_name})
    return {monitored_name}
"""

__all__ = ["compile_module"]


class Languange(str, Enum):
    c = "C"
    python = "Python"
    dolfin = "Dolfin"


def module_from_string(code, name="module"):
    module = types.ModuleType(name)
    exec(code, module.__dict__)
    return module


def compile_module(
    ode: typing.Union[ODE, str],
    language: Languange = Languange.c,
    monitored: typing.Optional[typing.List[str]] = None,
    generation_params: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> types.ModuleType:
    """JIT compile an ode

    Parameters
    ----------
    ode : typing.Union[ODE, str]
        The gotran ode
    language : Languange, optional
        The language of the generated code, by default Languange.c
    monitored : typing.Optional[typing.List[str]], optional
        A list of names of intermediates of the ODE. Code for monitoring
        the intermediates will be generated, by default None
    generation_params : typing.Optional[typing.Dict[str, typing.Any]], optional
        Parameters controling the code generation, by default None

    Returns
    -------
    types.ModuleType
        Module with ode functions
    """

    monitored = monitored or []
    generation_params = generation_params or {}

    check_arg(ode, (ODE, str))

    if isinstance(ode, str):
        ode = load_ode(ode)

    check_kwarg(language, "language", str)

    language = language.capitalize()
    valid_languages = ["C", "Python", "Dolfin"]
    if language not in valid_languages:
        value_error(
            "Expected one of {0} for the language kwarg.".format(
                ", ".join("'{0}'".format(lang) for lang in valid_languages),
            ),
        )

    params = parameters.generation.copy()
    params.update(generation_params)

    if language == "C":
        return compile_extension_module(
            ode,
            monitored,
            params,
        )

    # Create unique module name for this application run
    modulename = module_signature(ode, monitored, params, languange="python")

    # No module in cache generate python version
    if language == "Dolfin":
        pgen = DOLFINCodeGenerator(params)
    else:
        pgen = PythonCodeGenerator(params)

    # Generate class code, execute it and collect namespace
    code = "import numpy as np\nimport math" + pgen.class_code(ode, monitored=monitored)

    python_module = module_from_string(code, modulename)

    return getattr(python_module, class_name(ode.name))()


def add_dll_export(code_dict):
    d = {}
    for k, v in code_dict.items():
        d[k] = """extern "C" DLL_EXPORT""" + v
    return d


def parse_arguments(params):
    # Add function prototype
    args = []
    args_doc = []
    for arg in params.code.default_arguments:
        if arg == "s":
            args.append("states")
            args_doc.append(
                f"""    {params.code.states.array_name} : np.ndarray
        The state values""",
            )
        elif arg == "t":
            args.append("time")
            args_doc.append(
                """    time : scalar
        The present time""",
            )
        elif arg == "p" and params.code.parameters.representation != "numerals":
            args.append("parameters")
            args_doc.append(
                f"""    {params.code.parameters.array_name} : np.ndarray
        The parameter values""",
            )
        elif arg == "b" and params.code.body.representation != "named":
            args.append("body")
            args_doc.append(
                f"""    {params.code.body.array_name} : np.ndarray
        The body values""",
            )

    args = ", ".join(args)
    args_doc = "\n".join(args_doc)
    return args, args_doc


def parse_jacobian_declarations(
    ode,
    args,
    args_doc,
    params,
):

    jacobian_declaration = ""
    if not params.functions.jacobian.generate:
        return jacobian_declaration

    # Flatten jacobian params
    if not params.code.array.flatten:
        debug("Generating jacobian C-code, forcing jacobian array " "to be flattened.")
        params.code.array.flatten = True

    jacobian_declaration = jacobian_template.format(
        num_states=ode.num_full_states,
        args=args,
        args_doc=args_doc,
        jac_name=params.functions.jacobian.result_name,
        jacobian_function_name=params.functions.jacobian.function_name,
    )
    return jacobian_declaration


def parse_monitor_declaration(ode, args, args_doc, params, monitored):
    monitor_declaration = ""
    if monitored and params.functions.monitored.generate:
        monitor_declaration = monitor_template.format(
            num_states=ode.num_full_states,
            num_monitored=len(monitored),
            args=args,
            args_doc=args_doc,
            monitored_name=params.functions.monitored.result_name,
            monitored_function_name=params.functions.monitored.function_name,
        )
    return monitor_declaration


def signature(ode, monitored, params, languange) -> str:
    return hashlib.sha1(
        str(
            ode.signature()
            + str(monitored)
            + repr(params)
            + languange
            + __version__
            + cppyy_version,
        ).encode("utf-8"),
    ).hexdigest()


def module_signature(ode, monitored, params, languange):
    return "gotran_compiled_module_{0}_{1}".format(
        class_name(ode.name),
        signature(ode, monitored, params, languange),
    )


def compile_extension_module(
    ode,
    monitored,
    params,
):
    """
    Compile an extension module, based on the C code from the ode
    """
    if not has_cppyy():
        raise ImportError("Please install 'cppyy'")

    args, args_doc = parse_arguments(params)

    # Do not generate any Python functions
    python_params = params.copy()
    for name in python_params.functions:
        if name == "monitored":
            continue
        python_params.functions[name].generate = False

    jacobian_code = parse_jacobian_declarations(ode, args, args_doc, params)
    monitor_code = parse_monitor_declaration(ode, args, args_doc, params, monitored)

    pgen = PythonCodeGenerator(python_params)
    cgen = CppCodeGenerator(params)

    clsname = class_name(ode.name) + "_" + signature(ode, monitored, params, "Cpp")
    print(clsname)
    class_code = cgen.class_code(ode=ode, monitored=monitored, clsname=clsname)
    try:
        _cppyygbl = __import__("cppyy.gbl", fromlist=[clsname])
        getattr(_cppyygbl, clsname)()
    except AttributeError:
        cppyy.cppdef(class_code)

    pcode = "\n\n".join(list(pgen.code_dict(ode, monitored=monitored).values()))

    rhs_code = rhs_template.format(
        rhs_function_name=params.functions.rhs.function_name,
        args=args,
        rhs_name=params.functions.rhs.result_name,
        args_doc=args_doc,
        states_name=params.code.states.array_name,
    )

    compiled_module_code = module_template.format(
        clsname=clsname,
        code="\n\n\n".join([rhs_code, pcode, monitor_code, jacobian_code]),
    )

    mymodule = module_from_string(compiled_module_code, clsname)

    info("Calling GOTRAN just-in-time (JIT) compiler, this may take some " "time...")
    sys.stdout.flush()

    info(" done")
    sys.stdout.flush()

    return mymodule
