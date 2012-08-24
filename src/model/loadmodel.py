__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2010 " + __author__
__date__ = "2012-05-07 -- 2012-08-25"
__license__  = "GNU LGPL Version 3.0 or later"

__all__ = ["load_ode", "get_load_arguments", "get_load_namespace"]

# System imports
import os
import sympy

# modelparameters import
from modelparameters.parameters import ScalarParam, ArrayParam, ConstParam

# gotran imports
from gotran2.common import *
from ode import ODE
import operations

global _namespace, _load_arguments

_load_arguments = None
_namespace = None

def load_ode(filename, **kwargs):
    """
    Load an ODE from file and return the instance

    The method looks for a file with .ode extension.

    Arguments
    ---------
    filename : str
        Name of the ode file to load
    """

    global _namespace, _load_arguments

    _load_arguments = kwargs

    # If a TypeCheck is provided turn it into its value
    #for key, value in kwargs.items():
    #    if isinstance(value, TypeCheck):
    #        kwargs[key] = value.get()

    # Create an ODE which will be populated with data when ode file is loaded
    ode = ODE(filename)

    print filename

    debug("Loading {}".format(ode))

    # Create namespace which the ode file will be executed in
    _namespace = {}
    _namespace.update(operations.__dict__)
    _namespace.update(sympy.functions.__dict__)
    _namespace.update(dict(t=ode.t, dt=ode.dt,
                           ScalarParam=ScalarParam,
                           ArrayParam=ArrayParam,
                           ConstParam=ConstParam))

    # Execute the file
    if len(filename) < 4 or filename[-4:] != ".ode":
        filename = filename + ".ode"
    
    if (not os.path.isfile(filename)):
        error("Could not find '{0}'".format(filename))
    
    execfile(filename, _namespace, {})

    collected_info = []
    
    info("Loaded ODE model '{0}' with:".format(ode.name))
    for what in ["states", "parameters", "variables"]:
        num = getattr(ode, "num_{0}".format(what))
        if num:
            info("{0}: {1}".format(("Num "+what).rjust(15), num))
    
    return ode

def get_load_arguments():
    """
    Return the present load_arguments
    """

    global _load_arguments
    return _load_arguments

def get_load_namespace():
    """
    Return the present namespace
    """

    global _namespace
    return _namespace
