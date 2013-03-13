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

import sys
import os
import re
import numpy
import instant
import hashlib
import types

import gotran
from gotran.common import check_arg, check_kwarg, push_log_level, \
     pop_log_level, info, INFO, value_error
from gotran.model.ode import ODE
from gotran.model.loadmodel import load_ode
from oderepresentation import ODERepresentation, _default_params
from codegenerator import CodeGenerator, CCodeGenerator

# Create doc string to jit
_compile_module_doc_str = "\n".join("    {0} : bool\n       {1}".format(\
    param.name, param.description) for param in _default_params().values())

_compile_module_doc_str = """
    JIT compile an ode

    Arguments:
    ----------
    ode : ODE, ODERepresentation
       The gotran ode
    rhs_args : str (optional)
       Argument order of the generated rhs function. 
       s=states, p=parameters, t=time.
       Defaults : 'stp'
    language : str (optional)
       The language of the generated code
       Defaults : 'C' \xe2\x88\x88 ['C', 'Python'] 
{0}
    """.format(_compile_module_doc_str)

# Set log level of instant
instant.set_logging_level("WARNING")

__all__ = ["compile_module"]

_additional_declarations = r"""
%init%{{
import_array();
%}}

%include <exception.i>
%feature("autodoc", "1");

// Typemaps
%typemap(in) (double* dy)
{{
  // Check type
  if (!PyArray_Check($input))
    SWIG_exception(SWIG_TypeError, "Numpy array expected");

  // Get PyArrayObject
  PyArrayObject *xa = (PyArrayObject *)$input;

  // Check data type
  if (!(PyArray_ISCONTIGUOUS(xa) && PyArray_TYPE(xa) == NPY_DOUBLE))
    SWIG_exception(SWIG_TypeError, "Contigous numpy array of doubles "
                  "expected. Make sure the numpy array is contiguous, "
                  "and uses dtype=float_.");

  // Check size of passed array
  if ( PyArray_SIZE(xa) != {num_states} )
    SWIG_exception(SWIG_ValueError, "Expected a numpy array of size: "
                                  "{num_states}, for the dy argument.");
  
  $1 = (double *)PyArray_DATA(xa);
}}

%typemap(in) (const double* states)
{{
  // Check type
  if (!PyArray_Check($input))
    SWIG_exception(SWIG_TypeError, "Numpy array expected");

  // Get PyArrayObject
  PyArrayObject *xa = (PyArrayObject *)$input;

  // Check data type
  if (!(PyArray_ISCONTIGUOUS(xa) && PyArray_TYPE(xa) == NPY_DOUBLE))
    SWIG_exception(SWIG_TypeError, "Contigous numpy array of doubles expected."
           " Make sure the numpy array is contiguous, and uses dtype=float_.");

  // Check size of passed array
  if ( PyArray_SIZE(xa) != {num_states} )
    SWIG_exception(SWIG_ValueError, "Expected a numpy array of size: "
                              "{num_states}, for the states argument.");
  
  $1 = (double *)PyArray_DATA(xa);
}}

%typemap(in) (double* parameters)
{{
  // Check type
  if (!PyArray_Check($input))
    SWIG_exception(SWIG_TypeError, "Numpy array expected");

  // Get PyArrayObject
  PyArrayObject *xa = (PyArrayObject *)$input;

  // Check data type
  if (!(PyArray_ISCONTIGUOUS(xa) && PyArray_TYPE(xa) == NPY_DOUBLE))
    SWIG_exception(SWIG_TypeError, "Contigous numpy array of doubles expected."
           " Make sure the numpy array is contiguous, and uses dtype=float_.");

  // Check size of passed array
  if ( PyArray_SIZE(xa) != {num_parameters} )
    SWIG_exception(SWIG_ValueError, "Expected a numpy array of size: "
             "{num_parameters}, for the parameters argument.");
  
  $1 = (double *)PyArray_DATA(xa);
}}

// The typecheck
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) double *
{{
    $1 = PyArray_Check($input) ? 1 : 0;
}}

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) const double *
{{
    $1 = PyArray_Check($input) ? 1 : 0;
}}

%pythoncode%{{
def rhs({args}, dy=None):
    '''
    Evaluates the right hand side of the model

    Arguments:
    ----------
{args_doc}    
    dy : np.ndarray (optional)
        The computed result 
    '''
    import numpy as np
    if dy is None:
        dy = np.zeros_like(states)
    _rhs({args}, dy)
    return dy

{python_code}
%}}

// Rename rhs to _rhs
%rename (_rhs) rhs;

"""

def compile_module(ode, rhs_args="stp", language="C", **options):

    check_arg(ode, (ODERepresentation, ODE, str))

    if isinstance(ode, str):
        ode = load_ode(ode)

    if isinstance(ode, ODE):
        params = _default_params()
        params.update(options)
        oderepr = ODERepresentation(ode, **params.copy(True))
    else:
        oderepr = ode

    check_kwarg(rhs_args, "rhs_args" ,str)
    check_kwarg(language, "language" ,str)

    language = language.capitalize()
    valid_languages = ["C", "Python"]
    if language not in valid_languages:
        value_error("Expected one of {0} for the language kwarg.".format(\
            ", ".join("'{0}'".format(lang) for lang in valid_languages)))
        
    if language == "C":
        return compile_extension_module(oderepr, rhs_args)

    pgen = CodeGenerator(oderepr)

    # Generate class code, execute it and collect namespace
    ns = {}
    exec(pgen.class_code(rhs_args), {}, ns)

    # Grab the only name defined in the executed code
    return ns.values()[0]()

# Assign docstring
compile_module.func_doc = _compile_module_doc_str

def compile_extension_module(oderepr, rhs_args):
    """
    Compile an extension module, based on the C code from the ode
    """
    
    # Add function prototype
    args=[]
    args_doc=[]
    for arg in rhs_args:
        if arg == "s":
            args.append("states")
            args_doc.append("""    states : np.ndarray
        The state values""")
        elif arg == "t":
            args.append("time")
            args_doc.append("""    time : scalar
        The present time""")
        elif arg == "p" and \
                 not oderepr.optimization.parameter_numerals:
            args.append("parameters")
            args_doc.append("""    parameters : np.ndarray
        The parameter values""")

    args = ", ".join(args)
    args_doc = "\n".join(args_doc)

    pgen = CodeGenerator(oderepr)
    cgen = CCodeGenerator(oderepr)

    code = cgen.dy_code(rhs_args=rhs_args, \
                         parameters_in_signature=True)

    pcode = "\n\n" + pgen.init_states_code() + "\n\n" + \
            pgen.init_param_code() + "\n\n" + \
            pgen.state_name_to_index_code() + "\n\n" + \
            pgen.param_name_to_index_code()

    # Create unique module name for this application run
    module_name = "gotran_compiled_module_{0}_{1}".format(\
        oderepr.ode.name, hashlib.md5(repr(code) + instant.get_swig_version() + \
                                      gotran.__version__ + \
                                      instant.__version__).hexdigest())
    
    # Check cache
    compiled_module = instant.import_module(module_name)

    if compiled_module:
        return compiled_module

    push_log_level(INFO)
    info("Calling GOTRAN just-in-time (JIT) compiler, this may take some "\
         "time...")
    sys.stdout.flush()

    # Configure instant and add additional system headers
    instant_kwargs = configure_instant()

    declaration_form = dict(\
        num_states = oderepr.ode.num_states,
        num_parameters = oderepr.ode.num_parameters,
        python_code = pcode,
        args=args,
        args_doc=args_doc,
        )

    # Compile extension module with instant
    compiled_module = instant.build_module(\
        code              = code,
        additional_declarations = _additional_declarations.format(\
            **declaration_form),
        signature         = module_name,
        **instant_kwargs)

    info(" done")
    pop_log_level()
    sys.stdout.flush()
    return compiled_module

def configure_instant():
    """
    Check system requirements

    Returns a dict with kwargs that can be passed to instant.build_module.
    """
    instant_kwargs = {}
    swig_include_dirs = []

    instant_kwargs['swig_include_dirs'] = swig_include_dirs
    instant_kwargs['include_dirs'] = [numpy.get_include()]
    instant_kwargs['system_headers'] = ["numpy/arrayobject.h", "math.h"]
    instant_kwargs['swigargs'] =['-O -c++']
    instant_kwargs['cppargs'] = ['-O2']

    return instant_kwargs
