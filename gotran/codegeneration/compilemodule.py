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
from gotran.common.options import parameters
from gotran.codegeneration.codegenerators import PythonCodeGenerator, \
     CCodeGenerator, class_name
from gotran.codegeneration.algorithmcomponents import rhs_expressions, \
     monitored_expressions, jacobian_expressions

# Set log level of instant
instant.set_log_level("WARNING")

__all__ = ["compile_module"]

additional_declarations_ = r"""
%init%{{
import_array();
%}}

%include <exception.i>
%feature("autodoc", "1");

// Typemaps
%typemap(in) (double* {rhs_name})
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
                                  "{num_states}, for the {rhs_name} argument.");
  
  $1 = (double *)PyArray_DATA(xa);
}}

%typemap(in) (const double* {states_name})
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
                              "{num_states}, for the {states_name} argument.");
  
  $1 = (double *)PyArray_DATA(xa);
}}

%typemap(in) (double* {states_name})
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
                              "{num_states}, for the {states_name} argument.");
  
  $1 = (double *)PyArray_DATA(xa);
}}

%typemap(in) (double* {parameters_name})
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
             "{num_parameters}, for the {parameters_name} argument.");
  
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
def {rhs_function_name}({args},{rhs_name} = None):
    '''
    Evaluates the right hand side of the model

    Arguments:
    ----------
{args_doc}    
    {rhs_name} : np.ndarray (optional)
        The computed result 
    '''
    import numpy as np
    if {rhs_name} is None:
        {rhs_name} = np.zeros_like({states_name})

    _{rhs_function_name}({args}, {rhs_name})
    return {rhs_name}

{python_code}
%}}

%rename (_{rhs_function_name}) {rhs_function_name};

{jacobian_declaration}

{monitor_declaration}
"""

jacobian_declaration_template_ = """
// Typemaps
%typemap(in) (double* {jac_name})
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
  if ( PyArray_SIZE(xa) != {num_states}*{num_states} )
    SWIG_exception(SWIG_ValueError, "Expected a numpy array of size: "
                                  "{num_states}*{num_states}, for the {jac_name} argument.");
  
  $1 = (double *)PyArray_DATA(xa);
}}

%pythoncode%{{
def {jacobian_function_name}({args}, {jac_name}=None):
    '''
    Evaluates the jacobian of the model

    Arguments:
    ----------
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
    
    _{jacobian_function_name}({args}, {jac_name})
    {jac_name}.shape = ({num_states},{num_states})
    return {jac_name}
%}}

%rename (_{jacobian_function_name}) {jacobian_function_name};
"""

monitor_declaration_template = """
%typemap(in) (double* {monitored_name})
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
  if ( PyArray_SIZE(xa) != {num_monitored} )
    SWIG_exception(SWIG_ValueError, "Expected a numpy array of size: "
             "{num_monitored}, for the {monitored_name} argument.");
  
  $1 = (double *)PyArray_DATA(xa);
  
}}

%pythoncode%{{
def {monitored_function_name}({args}, {monitored_name}=None):
    '''
    Evaluates any monitored intermediates of the model

    Arguments:
    ----------
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
    
    _{monitored_function_name}({args}, {monitored_name})
    return {monitored_name}
%}}

%rename (_{monitored_function_name}) {monitored_function_name};
"""

def compile_module(ode, language="C", monitored=None,
                   generation_params=None,
                   additional_declarations=None,
                   jacobian_declaration_template=None):
    """
    JIT compile an ode
    
    Arguments:
    ----------
    ode : ODE, str
        The gotran ode
    language : str (optional)
        The language of the generated code
        Defaults : 'C' \xe2\x88\x88 ['C', 'Python'] 
    monitored : list
        A list of names of intermediates of the ODE. Code for monitoring
        the intermediates will be generated.
    generation_params : dict
        Parameters controling the code generation
    """
    
    monitored = monitored or []
    generation_params = generation_params or {}

    check_arg(ode, (ODE, str))
    
    if isinstance(ode, str):
        ode = load_ode(ode)
    
    check_kwarg(language, "language", str)

    language = language.capitalize()
    valid_languages = ["C", "Python"]
    if language not in valid_languages:
        value_error("Expected one of {0} for the language kwarg.".format(\
            ", ".join("'{0}'".format(lang) for lang in valid_languages)))
        
    params = parameters.generation.copy()
    params.update(generation_params)

    if language == "C":
        return compile_extension_module(ode, monitored, params,
                                        additional_declarations,
                                        jacobian_declaration_template)

    # Create unique module name for this application run
    modulename = "gotran_python_module_{0}_{1}".format(\
        class_name(ode.name), hashlib.sha1(str(\
            ode.signature() + str(monitored) + repr(params) + \
            gotran.__version__ + instant.__version__).encode('utf-8')).hexdigest())
    
    # Check cache
    python_module = instant.import_module(modulename)
    if python_module:
        return getattr(python_module, class_name(ode.name))()

    # No module in cache generate python version
    pgen = PythonCodeGenerator(params)

    # Generate class code, execute it and collect namespace
    code = "from __future__ import division\nimport numpy as np\nimport math" + pgen.class_code(ode, monitored=monitored)

    # Make a temporary module path for compilation
    module_path = os.path.join(instant.get_temp_dir(), modulename)
    instant.instant_assert(not os.path.exists(module_path),
                           "Not expecting module_path to exist: '{}'".format(module_path))
    instant.makedirs(module_path)
    original_path = os.getcwd()
    try:
        module_path = os.path.abspath(module_path)
        os.chdir(module_path)
        instant.write_file("__init__.py", code)
        module_path = instant.copy_to_cache(\
            module_path, instant.get_default_cache_dir(), modulename)
    
    finally:
        # Always get back to original directory.
        os.chdir(original_path)

    python_module = instant.import_module(modulename)
    return getattr(python_module, class_name(ode.name))()

def compile_extension_module(ode, monitored, params,
                             additional_declarations=None,
                             jacobian_declaration_template=None):
    """
    Compile an extension module, based on the C code from the ode
    """
    
    # Add function prototype
    args=[]
    args_doc=[]
    for arg in params.code.default_arguments:
        if arg == "s":
            args.append("states")
            args_doc.append("""    {0} : np.ndarray
        The state values""".format(params.code.states.array_name))
        elif arg == "t":
            args.append("time")
            args_doc.append("""    time : scalar
        The present time""")
        elif arg == "p" and \
                 params.code.parameters.representation != "numerals":
            args.append("parameters")
            args_doc.append("""    {0} : np.ndarray
        The parameter values""".format(params.code.parameters.array_name))
        elif arg == "b" and \
                 params.code.body.representation != "named":
            args.append("body")
            args_doc.append("""    {0} : np.ndarray
        The body values""".format(params.code.body.array_name))

    # Create unique module name for this application run
    modulename = "gotran_compiled_module_{0}_{1}".format(\
        class_name(ode.name), hashlib.sha1(str(\
            ode.signature() + str(monitored) + repr(params) + \
            gotran.__version__ + instant.__version__).encode('utf-8')).hexdigest())
    
    # Check cache
    compiled_module = instant.import_module(modulename)

    if compiled_module:
        return compiled_module

    args = ", ".join(args)
    args_doc = "\n".join(args_doc)

    # Do not generate any Python functions
    python_params = params.copy()
    for name in python_params.functions:
        if name == "monitored":
            continue
        python_params.functions[name].generate = False

    jacobian_declaration = ""
    monitor_declaration = ""
    if params.functions.jacobian.generate:

        # Flatten jacobian params
        if not params.code.array.flatten:
            debug("Generating jacobian C-code, forcing jacobian array "\
                  "to be flattened.")
            params.code.array.flatten = True

        jacobian_declaration_template = jacobian_declaration_template_ if \
                                        jacobian_declaration_template is None \
                                        else jacobian_declaration_template
        
        jacobian_declaration = jacobian_declaration_template.format(\
            num_states = ode.num_full_states,
            args=args,
            args_doc=args_doc,
            jac_name=params.functions.jacobian.result_name,
            jacobian_function_name=params.functions.jacobian.function_name,
            )

    if monitored and params.functions.monitored.generate:
        monitor_declaration = monitor_declaration_template.format(\
            num_states = ode.num_full_states,
            num_monitored = len(monitored),
            args=args,
            args_doc=args_doc,
            monitored_name=params.functions.monitored.result_name,
            monitored_function_name=params.functions.monitored.function_name,
            )
    
    pgen = PythonCodeGenerator(python_params)
    cgen = CCodeGenerator(params)
    
    pcode = "\n\n".join(\
        list(pgen.code_dict(ode, monitored=monitored).values()))
    
    ccode = "\n\n".join(list(cgen.code_dict(ode,
                        monitored=monitored,
                        include_init=False,
                        include_index_map=False).values()))
    
    # push_log_level(INFO)
    info("Calling GOTRAN just-in-time (JIT) compiler, this may take some "\
         "time...")
    sys.stdout.flush()

    # Configure instant and add additional system headers
    instant_kwargs = configure_instant()



    declaration_form = dict(\
        num_states = ode.num_full_states,
        num_parameters = ode.num_parameters,
        num_monitored = len(monitored),
        python_code = pcode,
        args=args,
        args_doc=args_doc,
        jacobian_declaration=jacobian_declaration,
        rhs_name=params.functions.rhs.result_name,
        rhs_function_name=params.functions.rhs.function_name,
        states_name=params.code.states.array_name,
        parameters_name=params.code.parameters.array_name,
        monitor_declaration=monitor_declaration,
        )

    additional_declarations = additional_declarations_ if \
                              additional_declarations is None \
                              else additional_declarations
    # Compile extension module with instant
    compiled_module = instant.build_module(\
        code  = ccode,
        additional_declarations = additional_declarations.format(\
            **declaration_form),
        signature = modulename,
        **instant_kwargs)

    info(" done")
    # pop_log_level()
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
    instant_kwargs['system_headers'] = ["numpy/arrayobject.h", "math.h"]#, "complex.h"]
    instant_kwargs['swigargs'] =['-O -c++']
    instant_kwargs['cppargs'] = ['-O2']

    return instant_kwargs
