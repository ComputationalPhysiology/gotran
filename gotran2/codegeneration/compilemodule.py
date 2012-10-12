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

import gotran2
from gotran2.common import check_arg, push_log_level, pop_log_level, info, INFO
from oderepresentation import ODERepresentation
from codegenerator import CCodeGenerator


# Set log level of instant
instant.set_logging_level("WARNING")

__all__ = ["jit"]

_additional_declarations = r"""
%module {module_name}

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

"""

def jit(oderepr):
    """
    JIT compile an ode 
    """

    # FIXME: Very rudimentary implementation
    # FIXME: Make more versatil
    check_arg(oderepr, ODERepresentation)

    gen = CCodeGenerator(oderepr)

    code = gen.dy_code(parameters_in_signature=True)
    
    # Compile module
    return compile_extension_module(code, oderepr.ode)

def compile_extension_module(code, ode):
    """
    Compile an extension module, based on the C code from the ode
    """
    
    # Create unique module name for this application run
    module_name = "gotran_compiled_module_{0}_{1}".format(\
        ode.name, hashlib.md5(repr(code) + instant.get_swig_version() + \
                              gotran2.__version__).hexdigest())
        
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
        module_name = ode.name,
        num_states = ode.num_states,
        num_parameters = ode.num_parameters,
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


