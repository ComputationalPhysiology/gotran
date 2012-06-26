__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2010 " + __author__
__date__ = "2012-05-07 -- 2012-05-08"
__license__  = "GNU LGPL Version 3.0 or later"

__all__ = ["load_ode", "get_load_arguments", "get_load_namespace"]

# System imports
import os

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

    @type filename : str
    @param filename : Name of the ode file to look for
    """

    global _namespace, _load_arguments

    _load_arguments = kwargs

    # If a TypeCheck is provided turn it into its value
    #for key, value in kwargs.items():
    #    if isinstance(value, TypeCheck):
    #        kwargs[key] = value.get()

    # Create an ODE which will be populated with data when ode file is loaded
    ode = ODE(filename)

    debug("Loading {}".format(ode))

    # Create namespace which the ode file will be executed in
    _namespace = {}
    _namespace.update(operations.__dict__)

    # Execute the file
    if len(filename) < 4 or filename[-4:] != ".ode":
        filename = filename + ".ode"
    
    if (not os.path.isfile(filename)):
        error("Could not find '{0}'".format(filename))
    
    execfile(filename, _namespace, {})

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
