# Copyright (C) 2011-2012 Johan Hake
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

__all__ = ["load_ode", "exec_ode"]

# System imports
import os
from collections import OrderedDict

# modelparameters import
from modelparameters.parameters import Param, ScalarParam, ArrayParam, \
     ConstParam, scalars
from modelparameters.sympytools import sp_namespace, sp, ModelSymbol

# gotran imports
from gotran.common import *
from gotran.model.ode import ODE

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
        
        # Get ODE
        ode = _get_load_ode()

        # Set the attr of the ODE
        if isinstance(value, scalars) or (isinstance(value, sp.Basic) and \
                                          any(isinstance(atom, ModelSymbol)\
                                              for atom in value.atoms())):

            # Set attribute
            setattr(ode, name, value)
        
            # Populate the name space with symbol attribute
            dict.__setitem__(self, name, getattr(ode, name))
        else:

            # If no ode attr was generated we just add the value to the
            # namespace
            dict.__setitem__(self, name, value)

def _init_namespace(ode):
    # Get global variables and reset them
    namespace = _get_load_namespace()
    namespace.clear()
    namespace.update(sp_namespace)
    namespace.update(dict(time=ode.time, dt=ode.dt,
                          ScalarParam=ScalarParam,
                          ArrayParam=ArrayParam,
                          ConstParam=ConstParam,
                          states=_states,
                          parameters=_parameters,
                          variables=_variables,
                          diff=ode.diff,
                          comment=ode.add_comment,
                          component=ode.set_component,
                          monitor=ode.add_monitored,
                          subode=ode.add_subode,
                          sp=sp,
                          model_arguments=_model_arguments))
    return namespace

def _reset_globals():
    _get_load_arguments().clear()
    _get_load_namespace().clear()
    _set_load_ode(None)

def exec_ode(ode_str, name):
    """
    Execute an ode given by a str
    """
    # Create an ODE which will be populated with data when ode file is loaded
    ode = ODE(name)
    _set_load_ode(ode)

    debug("Loading {}".format(_current_ode))

    # Create namespace which the ode file will be executed in
    namespace = _init_namespace(ode)

    # Dict to collect declared intermediates
    intermediate_dispatcher = IntermediateDispatcher()

    # Execute file and collect 
    exec(ode_str, namespace, intermediate_dispatcher)
    
    # Check for completeness
    if not ode.is_complete:
        warning("ODE mode '{0}' is not complete.".format(ode.name))
    
    info("Loaded ODE model '{0}' with:".format(ode.name))
    for what in ["states", "parameters", "variables"]:
        num = getattr(ode, "num_{0}".format(what))
        if num:
            info("{0}: {1}".format(("Num "+what).rjust(22), num))

    # Reset global variables
    _reset_globals()
    
    return ode

def load_ode(filename, name=None, **kwargs):
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

    timer = Timer("Load ODE")

    # If a Param is provided turn it into its value
    for key, value in kwargs.items():
        if isinstance(value, Param):
            kwargs[key] = value.getvalue()
    
    arguments = _get_load_arguments()
    arguments.clear()
    arguments.update(kwargs)

    # Extract name from filename
    if len(filename) < 4 or filename[-4:] != ".ode":
        name = name or filename
        filename = filename + ".ode"
    elif name is None:
        name = filename[:-4]

    # Create an ODE which will be populated with data when ode file is loaded
    ode = ODE(name)
    _set_load_ode(ode)

    debug("Loading {}".format(_current_ode))

    # Create namespace which the ode file will be executed in
    _init_namespace(ode)

    # Execute the file
    if (not os.path.isfile(filename)):
        error("Could not find '{0}'".format(filename))

    # Dict to collect declared intermediates
    intermediate_dispatcher = IntermediateDispatcher()

    # Execute file and collect 
    execfile(filename, _namespace, intermediate_dispatcher)
    
    # Check for completeness
    if not ode.is_complete:
        warning("ODE mode '{0}' is not complete.".format(ode.name))
    
    info("Loaded ODE model '{0}' with:".format(ode.name))
    for what in ["states", "field_states", "parameters", "field_parameters",
                 "variables", "monitored_intermediates"]:
        num = getattr(ode, "num_{0}".format(what))
        if num:
            info("{0}: {1}".format(("Num "+what.replace("_", \
                                                        " ")).rjust(22), num))

    # Reset global variables
    _reset_globals()
    
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

def _states(component="", **kwargs):
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
    _add_entities(component, kwargs, "state")

def _parameters(component="", **kwargs):
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
    _add_entities(component, kwargs, "parameter")
    
def _variables(component="", **kwargs):
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
    _add_entities(component, kwargs, "variable")

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
    
def _add_entities(component, kwargs, entity):
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
        sym = add(name, value, component=component)
        
        # Add symbol to caller frames namespace
        try:
            debug("Adding {0} '{1}' to namespace".format(entity, name))
            if name in namespace:
                warning("Symbol with name: '{0}' already in namespace.".\
                        format(name))
            namespace[name] = sym
        except:
            error("Not able to add '{0}' to namespace".format(name))
