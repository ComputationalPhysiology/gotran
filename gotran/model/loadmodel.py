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
import inspect
import os
import re
import tempfile
from collections import OrderedDict

# modelparameters import
from modelparameters.parameters import Param, ScalarParam, ArrayParam, \
     ConstParam, scalars
from modelparameters.sympytools import sp_namespace, sp, ModelSymbol

# gotran imports
from gotran.common import *
from gotran.model.ode import ODE

_for_template = re.compile("for.*in .*:")
_no_intermediate_template = re.compile(".*# NO INTERMEDIATE.*")

class IntermediateDispatcher(dict):
    """
    Dispatch intermediates to ODE attributes
    """
    
    def __init__(self, ode):
        """
        Initalize with an ode
        """
        from ode import ODE
        check_arg(ode, ODE)
        self._ode = ode

    def __setitem__(self, name, value):

        # Set the attr of the ODE
        # If a scalar or a sympy number or it is a sympy.Basic consisting of
        # ModelSymbols
        if isinstance(value, scalars) or isinstance(value, sp.Number) or \
               (isinstance(value, sp.Basic) and \
                any(isinstance(atom, ModelSymbol)\
                    for atom in value.atoms())):

            # Get source which triggers the insertion to the global namespace
            frame = inspect.currentframe().f_back
            lines, lnum = inspect.findsource(frame)

            # Try getting the code
            try:
                code = lines[frame.f_lineno-1].strip()

                # Concatenate lines with line continuation symbols
                prev = 2
                while frame.f_lineno-prev >=0 and \
                          len(lines[frame.f_lineno-prev]) >= 2 and \
                          lines[frame.f_lineno-prev][-2:] == "\\\n":
                    code = lines[frame.f_lineno-prev][:-2].strip() + code
                    prev +=1
            except :
                code = ""
            
            # Check if the line includes a for statement
            # Here we strip op potiential code comments after the main for
            # statement.
            if re.search(_for_template, code.split("#")[0].strip()) or \
                   re.search(_no_intermediate_template, code):

                debug("Not registering '{0}' as an intermediate.".format(name))
                
                # If so just add the value to the namespace without
                # registering the intermediate
                dict.__setitem__(self, name, value)
                
            else:
                
                # Add intermediate
                sym = self._ode.add_intermediate(name, value)
        
                # Populate the name space with symbol attribute
                dict.__setitem__(self, name, sym)
        else:
            debug("Not registering '{0}' as an intermediate.".format(name))

            # If no ode attr was generated we just add the value to the
            # namespace
            dict.__setitem__(self, name, value)

def _init_namespace(ode, load_arguments):
    """
    Create namespace and populate it
    """
    
    namespace = {}
    namespace.update(sp_namespace)
    namespace.update(dict(time=ode.time, dt=ode.dt,
                          ScalarParam=ScalarParam,
                          ArrayParam=ArrayParam,
                          ConstParam=ConstParam,
                          diff=ode.diff,
                          comment=ode.add_comment,
                          component=ode.set_component,
                          monitor=ode.add_monitored,
                          sp=sp,
                          ))

    # Add ode and model_arguments
    _namespace_binder(namespace, ode, load_arguments)
    return namespace

def exec_ode(ode_str, name):
    """
    Execute an ode given by a str
    """
    # Create an ODE which will be populated with data when ode file is loaded
    ode = ODE(name, return_namespace=True)

    debug("Loading {}".format(ode.name))

    # Create namespace which the ode file will be executed in
    namespace = _init_namespace(ode, {})

    # Dict to collect declared intermediates
    intermediate_dispatcher = IntermediateDispatcher(ode)

    # Write str to file
    open("_tmp_gotrand.ode", "w").write(ode_str)
    
    # Execute file and collect 
    execfile("_tmp_gotrand.ode", namespace, intermediate_dispatcher)
    os.unlink("_tmp_gotrand.ode")

    # Finalize ODE
    ode.finalize()
    
    # Check for completeness
    if not ode.is_complete:
        warning("ODE mode '{0}' is not complete.".format(ode.name))
    
    info("Loaded ODE model '{0}' with:".format(ode.name))
    for what in ["states", "parameters", "variables"]:
        num = getattr(ode, "num_{0}".format(what))
        if num:
            info("{0}: {1}".format(("Num "+what).rjust(22), num))

    return ode

def load_ode(filename, name=None, **arguments):
    """
    Load an ODE from file and return the instance

    The method looks for a file with .ode extension.

    Arguments
    ---------
    filename : str
        Name of the ode file to load
    name : str (optional)
        Set the name of ODE (defaults to filename)
    arguments : dict (optional)
        Optional arguments which can control loading of model
    """

    timer = Timer("Load ODE")

    # If a Param is provided turn it into its value
    for key, value in arguments.items():
        if isinstance(value, Param):
            arguments[key] = value.getvalue()
    
    # Extract name from filename
    if len(filename) < 4 or filename[-4:] != ".ode":
        name = name or filename
        filename = filename + ".ode"
    elif name is None:
        name = filename[:-4]

    # Create an ODE which will be populated with data when ode file is loaded
    ode = ODE(name, return_namespace=True)

    debug("Loading {}".format(ode.name))

    # Create namespace which the ode file will be executed in
    namespace = _init_namespace(ode, arguments)

    # Execute the file
    if (not os.path.isfile(filename)):
        error("Could not find '{0}'".format(filename))

    # Dict to collect declared intermediates
    intermediate_dispatcher = IntermediateDispatcher(ode)

    # Execute file and collect 
    execfile(filename, namespace, intermediate_dispatcher)
    
    # Finalize ODE
    ode.finalize()
    
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
    return ode

def _namespace_binder(namespace, ode, load_arguments):
    """
    Add functions all bound to current ode, namespace and arguments
    """

    def subode(subode, prefix=None, components=None):
        """
        Load an ODE and add it to the present ODE

        Argument
        --------
        subode : str
            The subode which should be added.
        prefix : str (optional)
            A prefix which all state and parameters are prefixed with. If not
            given the name of the subode will be used as prefix. If set to
            empty string, no prefix will be used.
        components : list, tuple of str (optional)
            A list of components which will be extracted and added to the present
            ODE. If not given the whole ODE will be added.
        """
        from gotran.model.expressions import Expression
        
        check_arg(subode, str, 0)

        # Add the subode and update namespace
        namespace.update(ode.add_subode(subode, prefix=prefix, \
                                        components=components))
    
    def states(component="", **kwargs):
        """
        Add a number of states to the current ODE
    
        Example
        -------
    
        >>> ODE("MyOde")
        >>> states(e=0.0, g=1.0)
        """
    
        if not kwargs:
            error("expected at least one state")
        
        # Add the states and update namespace
        namespace.update(ode.add_states(component, **kwargs))
    
    def parameters(component="", **kwargs):
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
        
        # Add the parameters and update namespace
        namespace.update(ode.add_parameters(component, **kwargs))
        
    def variables(component="", **kwargs):
        """
        Add a number of variables to the current ODE
    
        Example
        -------
    
        >>> ODE("MyOde")
        >>> variables(c_out=0.0, c_in=1.0)
        
        """
    
        if not kwargs:
            error("expected at least one variable")
        
        # Add the variables and update namespace
        namespace.update(ode.add_variables(component, **kwargs))
    
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
    
        # Check the passed load arguments
        for key in load_arguments:
            if key not in kwargs:
                error("Name '{0}' is not a model_argument.".format(key))
    
        # Update the namespace
        for key, value in kwargs.items():
            if key not in load_arguments:
                namespace[key] = value
            else:
                namespace[key] = load_arguments[key]

    def markov_model(name, component="", algebraic_sum=None, **states):
        """        
        Initalize a Markov model

        Arguments
        ---------
        name : str
            Name of Markov model
        component : str (optional)
            Add state to a particular component
        algebraic_sum : scalar (optional)
            If the algebraic sum of all states should be constant,
            give the value here.
        states : dict
            A dict with all states defined in this Markov model
        """

        # Create Markov model
        mm = ode.add_markov_model(name, component=component, \
                                  algebraic_sum=algebraic_sum, \
                                  **states)
        
        # Add Markov model to namespace
        namespace[mm.name] = mm

        # Add symbols to namespace
        for state in mm._states:
            namespace[state.name] = state.sym

    # Update provided namespace
    namespace.update(dict(
        states=states,
        parameters=parameters,
        variables=variables,
        model_arguments=model_arguments,
        subode=subode,
        markov_model=markov_model,
        )
                         )
                    
