__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2010 " + __author__
__date__ = "2012-05-07 -- 2012-09-03"
__license__  = "GNU LGPL Version 3.0 or later"

__all__ = ["load_ode"]

# System imports
import os
from collections import OrderedDict

# modelparameters import
from modelparameters.parameters import ScalarParam, ArrayParam, ConstParam
from modelparameters.sympytools import sp_namespace

# gotran imports
from gotran2.common import *
from gotran2.model.ode import ODE

# Global variables
global _namespace, _load_arguments, _current_ode

_load_arguments = {}
_namespace = {}
_current_ode = None

class IntermediateDispatcher(dict):
    """
    Dispatch intermediates to ODE attributes
    """
    def __setitem__(self, name, value):
        
        ode = _get_load_ode()
        setattr(ode, name, value)
        dict.__setitem__(self, name, getattr(ode, name))

def load_ode(filename, name=None, collect_intermediates=True, **kwargs):
    """
    Load an ODE from file and return the instance

    The method looks for a file with .ode extension.

    Arguments
    ---------
    filename : str
        Name of the ode file to load
    name : str (optional)
        Set the name of ODE (defaults to filename)
    """

    # If a Param is provided turn it into its value
    for key, value in kwargs.items():
        if isinstance(value, Param):
            kwargs[key] = value.getvalue()

    # Get global variables and reset them
    namespace = _get_load_namespace()
    namespace.clear()
    
    arguments = _get_load_arguments()
    arguments.clear()
    arguments.update(kwargs)

    # Create an ODE which will be populated with data when ode file is loaded
    ode = ODE(name or filename)
    _set_load_ode(ode)

    debug("Loading {}".format(_current_ode))

    # Create namespace which the ode file will be executed in
    namespace.update(sp_namespace)
    namespace.update(dict(t=ode.t, dt=ode.dt,
                          ScalarParam=ScalarParam,
                          ArrayParam=ArrayParam,
                          ConstParam=ConstParam,
                          states=_states,
                          parameters=_parameters,
                          variables=_variables,
                          diff=ode.diff,
                          comment=ode.add_comment,
                          model_arguments=_model_arguments))

    # Execute the file
    if len(filename) < 4 or filename[-4:] != ".ode":
        filename = filename + ".ode"
    
    if (not os.path.isfile(filename)):
        error("Could not find '{0}'".format(filename))

    # Dict to collect declared intermediates
    intermediate_dispatcher = IntermediateDispatcher()

    # Execute file and collect 
    execfile(filename, _namespace, intermediate_dispatcher)

    info("Loaded ODE model '{0}' with:".format(ode.name))
    for what in ["states", "parameters", "variables"]:
        num = getattr(ode, "num_{0}".format(what))
        if num:
            info("{0}: {1}".format(("Num "+what).rjust(15), num))

    # Reset global variables
    _set_load_ode(None)
    namespace.clear()
    arguments.clear()
    
    return ode

def _get_load_arguments():
    """
    Return the present load_arguments
    """

    global _load_arguments
    return _load_arguments

def _get_load_namespace():
    """
    Return the present namespace
    """

    global _namespace
    return _namespace

def _get_load_ode():
    """
    Return the present ODE
    """

    global _current_ode
    return _current_ode

def _set_load_ode(ode):
    """
    Return the present ODE
    """

    global _current_ode
    _current_ode = ode

def _states(comment="", **kwargs):
    """
    Add a number of states to the current ODE

    Example
    -------

    >>> ODE("MyOde")
    >>> states(e=0.0, g=1.0)
    """

    if not kwargs:
        error("expected at least one state")
    
    # Check values and create sympy Symbols
    _add_entities(comment, kwargs, "state")

def _parameters(comment="", **kwargs):
    """
    Add a number of parameters to the current ODE

    Example
    -------

    >>> ODE("MyOde")
    >>> parameters(v_rest=-85.0,
                   v_peak=35.0,
                   time_constant=1.0)
    """
    
    if not kwargs:
        error("expected at least one state")
    
    # Check values and create sympy Symbols
    _add_entities(comment, kwargs, "parameter")
    
def _variables(comment="", **kwargs):
    """
    Add a number of variables to the current ODE

    Example
    -------

    >>> ODE("MyOde")
    >>> variables(c_out=0.0, c_in=1.0)
    
    """

    if not kwargs:
        error("expected at least one variable")
    
    # Check values and create sympy Symbols
    _add_entities(comment, kwargs, "variable")

def _model_arguments(**kwargs):
    """
    Defines arguments that can be altered while the ODE is loaded
    
    Example
    -------
    
    In gotran model file:

      >>> ...
      >>> model_arguments(include_Na=True)
      >>> if include_Na:
      >>>     states(Na=1.0)
      >>> ...

    When the model gets loaded
    
      >>> ...
      >>> load_model("model", include_Na=False)
      >>> ...
      
    """

    # Update namespace with load_arguments and model_arguments
    arguments = _get_load_arguments()
    namespace = _get_load_namespace()

    # Check the passed load arguments
    for key in arguments:
        if key not in kwargs:
            error("Name '{0}' is not a model_argument.".format(key))

    # Update the namespace
    for key, value in kwargs.items():
        if key not in arguments:
            namespace[key] = value
        else:
            namespace[key] = arguments[key]
    
def _add_entities(comment, kwargs, entity):
    """
    Help function for determine if each entity in the kwargs is unique
    and to check the type of the given default value
    """
    assert(entity in ["state", "parameter", "variable"])

    # Get current ode
    ode = _get_load_ode()

    # Get caller frame
    namespace = _get_load_namespace()

    # Get add method
    add = getattr(ode, "add_{0}".format(entity))
    
    # Symbol and value dicts
    for name in sorted(kwargs.keys()):

        # Get value
        value = kwargs[name]

        # Add the symbol
        sym = add(name, value, comment)
        
        # Add symbol to caller frames namespace
        try:
            debug("Adding {0} '{1}' to namespace".format(entity, name))
            if name in namespace:
                warning("Symbol with name: '{0}' already in namespace.".\
                        format(name))
            namespace[name] = sym
        except:
            error("Not able to add '{0}' to namespace".format(name))


