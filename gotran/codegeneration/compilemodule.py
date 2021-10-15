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
import importlib.util
import sys
import types
import typing
from enum import Enum
from pathlib import Path

import dijitso
from modelparameters.logger import debug
from modelparameters.logger import info
from modelparameters.logger import value_error
from modelparameters.utils import check_arg
from modelparameters.utils import check_kwarg

from .. import __version__
from ..common import GotranException
from ..common.options import parameters
from ..model.loadmodel import load_ode
from ..model.ode import ODE
from .codegenerators import CCodeGenerator
from .codegenerators import class_name
from .codegenerators import DOLFINCodeGenerator
from .codegenerators import PythonCodeGenerator

module_template = """import dijitso as _dijitso
import numpy as _np
import ctypes as _ctypes
_module = _dijitso.cache.load_library("{signature}", {cache})

_float64_array = _np.ctypeslib.ndpointer(
    dtype=_ctypes.c_double, ndim=1, flags="contiguous"
)

{bindings}

{code}
"""

binding_template = """_module.{funcname}.argtypes = [{argtypes}]
_module.{funcname}.restypes = {restype}
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

    # Check cache
    python_module = load_module(modulename)
    if python_module is not None:
        return getattr(python_module, class_name(ode.name))()

    # No module in cache generate python version
    if language == "Dolfin":
        pgen = DOLFINCodeGenerator(params)
    else:
        pgen = PythonCodeGenerator(params)

    # Generate class code, execute it and collect namespace
    code = "import numpy as np\nimport math" + pgen.class_code(ode, monitored=monitored)

    save_module(code, modulename)

    python_module = load_module(modulename)
    return getattr(python_module, class_name(ode.name))()


def _jit_generate(class_data, module_name, signature, parameters):
    """Helper function for ditjitso"""

    template_code = """
{includes}
// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

{code}
"""
    code_c = template_code.format(
        code="\n\n".join(list(class_data["code_dict"].values())),
        includes="\n".join(class_data["includes"]),
    )
    code_h = ""
    depends = []

    return code_h, code_c, depends


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


def args_to_argtypes(args: str) -> str:
    argtypes = {
        "states": "_float64_array",
        "time": "_ctypes.c_double",
        "parameters": "_float64_array",
        "values": "_float64_array",
    }
    args_split = args.split(", ")

    if len(args_split) == 3:
        # Values are optional to it is not in the list
        args_split.append("values")
    if len(args_split) != 4:
        raise GotranException(f"Expected number to arguments to be 4, got {args_split}")
    return ", ".join([argtypes[arg] for arg in args_split])


def cache_path(signature, cache_dir=None) -> Path:
    if cache_dir is None:
        dijitso_params = dijitso.validate_params(dijitso.params.default_params())
        cache_dir = dijitso_params["cache"]["cache_dir"]
    return Path(cache_dir).joinpath(f"{signature}.py")


def load_module(signature: str, cache_dir=None):

    path = cache_path(signature, cache_dir)
    if not path.is_file():
        return None

    spec = importlib.util.spec_from_file_location(signature, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except AttributeError:
        return None
    return module


def module_signature(ode, monitored, params, languange):
    return "gotran_compiled_module_{0}_{1}".format(
        class_name(ode.name),
        hashlib.sha1(
            str(
                ode.signature()
                + str(monitored)
                + repr(params)
                + languange
                + __version__
                + dijitso.__version__,
            ).encode("utf-8"),
        ).hexdigest(),
    )


def compile_extension_module(
    ode,
    monitored,
    params,
):
    """
    Compile an extension module, based on the C code from the ode
    """
    dijitso_params = dijitso.validate_params(dijitso.params.default_params())

    args, args_doc = parse_arguments(params)

    # Create unique module name for this application run
    modulename = module_signature(ode, monitored, params, languange="C")

    compiled_module = load_module(modulename)
    if compiled_module is not None:
        return compiled_module

    # Do not generate any Python functions
    python_params = params.copy()
    for name in python_params.functions:
        if name == "monitored":
            continue
        python_params.functions[name].generate = False

    jacobian_code = parse_jacobian_declarations(ode, args, args_doc, params)
    monitor_code = parse_monitor_declaration(ode, args, args_doc, params, monitored)

    pgen = PythonCodeGenerator(python_params)
    cgen = CCodeGenerator(params)
    code_dict = cgen.code_dict(
        ode,
        monitored=monitored,
        include_init=False,
        include_index_map=False,
    )

    pcode = "\n\n".join(list(pgen.code_dict(ode, monitored=monitored).values()))
    argtypes = args_to_argtypes(args)
    cpp_data = {
        "code_dict": add_dll_export(code_dict),
        "includes": ["#include <math.h>"],
    }

    rhs_code = rhs_template.format(
        rhs_function_name=params.functions.rhs.function_name,
        args=args,
        rhs_name=params.functions.rhs.result_name,
        args_doc=args_doc,
        states_name=params.code.states.array_name,
    )
    rhs_binding = binding_template.format(
        funcname="rhs",
        argtypes=argtypes,
        restype="None",
    )

    monitor_binding = ""
    if monitor_code != "":
        monitor_binding = binding_template.format(
            funcname="monitor",
            argtypes=argtypes,
            restype="None",
        )

    jacobian_binding = ""
    if jacobian_code != "":
        jacobian_binding = binding_template.format(
            funcname="compute_jacobian",
            argtypes=argtypes,
            restype="None",
        )

    compiled_module_code = module_template.format(
        signature=modulename,
        cache=repr(dijitso_params["cache"]),
        code="\n\n\n".join([rhs_code, pcode, monitor_code, jacobian_code]),
        bindings="\n".join([rhs_binding, monitor_binding, jacobian_binding]),
    )

    module, signature = dijitso.jit(
        cpp_data,
        modulename,
        dijitso_params,
        generate=_jit_generate,
    )

    save_module(compiled_module_code, modulename)

    info("Calling GOTRAN just-in-time (JIT) compiler, this may take some " "time...")
    sys.stdout.flush()

    info(" done")
    sys.stdout.flush()

    return load_module(signature)


def save_module(
    code: str,
    signature: str,
    cache_dir: typing.Optional[str] = None,
) -> None:
    with open(
        cache_path(signature=signature, cache_dir=cache_dir),
        "w",
    ) as f:
        f.write(code)
