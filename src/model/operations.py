__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2010 " + __author__
__date__ = "2012-02-22 -- 2012-08-31"
__license__  = "GNU LGPL Version 3.0 or later"

__all__ = ["states", "parameters", "diff", "variables", "model_arguments"]

# System imports
import inspect
import types

# Gotran imports
from gotran2.common import *
from gotran2.model.ode import gco

def states(comment="", **kwargs):
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

def parameters(comment="", **kwargs):
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
    
def variables(comment="", **kwargs):
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

def diff(derivatives, expr):
    """
    Set derivative of a declared state

    Arguments
    ---------
    derivatives : State, list of States or 0
        If derivatives is a single state then it is interpreted as an ODE
        If a list of states (with possible scalar weights) or 0 is
        given, it is interpreted as a DAE expression.
    expr : Sympy expression of ModelSymbols
        The derivative expression
    """
    
    # Get current ode
    ode = gco()

    ode.diff(derivatives, expr)

def model_arguments(**kwargs):
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

    # Import here to avoid circular dependencies
    from loadmodel import get_load_arguments, get_load_namespace
    
    # Update namespace with load_arguments and model_arguments
    arguments = get_load_arguments()
    namespace = get_load_namespace()

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
    ode = gco()

    # Get caller frame
    frame = inspect.currentframe().f_back.f_back

    # Get add method
    add = getattr(ode, "add_{0}".format(entity))
    
    # Symbol and value dicts
    for name, value in kwargs.iteritems():

        # Add the symbol
        sym = add(name, value, comment)
        
        # Add symbol to caller frames namespace
        try:
            debug("Adding {0} '{1}' to namespace".format(entity, name))
            frame.f_globals[name] = sym
        except:
            error("Not able to add '{0}' to namespace".format(name))


