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
from collections import OrderedDict, deque

# modelparameters import
from modelparameters.parameters import Param, ScalarParam, ArrayParam, \
     ConstParam, scalars
from modelparameters.sympytools import sp_namespace, sp, symbols_from_expr

# gotran imports
from gotran.common import *
from gotran.model.odecomponents2 import ODE

_for_template = re.compile("\A[ ]*for[ ]+.*in[ ]+:[ ]*\Z")
_no_intermediate_template = re.compile(".*# NO INTERMEDIATE.*")

class IntermediateDispatcher(dict):
    """
    Dispatch intermediates to ODE attributes
    """
    
    def __init__(self, ode=None):
        """
        Initalize with an ode
        """
        self.ode = ode

    def __setitem__(self, name, value):

        # Set the attr of the ODE
        # If a scalar or a sympy number or it is a sympy.Basic consisting of
        # sp.Symbols

        if isinstance(value, scalars) or isinstance(value, sp.Number) or \
               (isinstance(value, sp.Basic) and symbols_from_expr(value)):

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

                # Add obj to the present component
                sym = setattr(self.ode.present_component, name, value)
        
        else:
            debug("Not registering '{0}' as an intermediate.".format(name))

            # If no ode attr was generated we just add the value to the
            # namespace
            dict.__setitem__(self, name, value)

    def update(self, other):
        check_arg(other, dict)
        for name, value in other.items():
            dict.__setitem__(self, name, value)

def _init_namespace(ode, load_arguments, namespace):
    """
    Create namespace and populate it
    """
    
    namespace.update(sp_namespace)

    # Add Sympy matrix related stuff
    namespace.update(dict(eye=sp.eye, diag=sp.diag, Matrix=sp.Matrix, zeros=sp.zeros))
                     
    namespace.update(dict(t=ode.t,
                          ScalarParam=ScalarParam,
                          ArrayParam=ArrayParam,
                          ConstParam=ConstParam,
                          comment=ode.add_comment,
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

    # Dict to collect declared intermediates
    intermediate_dispatcher = IntermediateDispatcher(ode)

    # Create namespace which the ode file will be executed in
    namespace = _init_namespace(ode, {}, intermediate_dispatcher)

    # Write str to file
    open("_tmp_gotrand.ode", "w").write(ode_str)
    
    # Execute file and collect 
    execfile("_tmp_gotrand.ode", intermediate_dispatcher)
    os.unlink("_tmp_gotrand.ode")

    # Finalize ODE
    ode.finalize()
    
    # Check for completeness
    if not ode.is_complete:
        warning("ODE mode '{0}' is not complete.".format(ode.name))
    
    info("Loaded ODE model '{0}' with:".format(ode.name))
    for what in ["states", "parameters"]:
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

    # Dict to collect namespace
    intermediate_dispatcher = IntermediateDispatcher()

    # Create an ODE which will be populated with data when ode file is loaded
    ode = ODE(name, intermediate_dispatcher)
    intermediate_dispatcher.ode = ode

    debug("Loading {}".format(ode.name))

    # Create namespace which the ode file will be executed in
    namespace = _init_namespace(ode, arguments, intermediate_dispatcher)

    # Execute the file
    if (not os.path.isfile(filename)):
        error("Could not find '{0}'".format(filename))

    # Execute file and collect
    execfile(filename, intermediate_dispatcher)
    
    # Finalize ODE
    ode.finalize()
    
    # Check for completeness
    if not ode.is_complete:
        warning("ODE mode '{0}' is not complete.".format(ode.name))
    
    info("Loaded ODE model '{0}' with:".format(ode.name))
    for what in ["states", "field_states", "parameters", "field_parameters",
                 "monitored_intermediates"]:
        num = getattr(ode, "num_{0}".format(what))
        if num:
            info("{0}: {1}".format(("Num "+what.replace("_", \
                                                        " ")).rjust(25), num))
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
        
        check_arg(subode, str, 0)

        # Add the subode and update namespace
        namespace.update(ode.add_subode(subode, prefix=prefix, \
                                        components=components))
    
    def states(*args, **kwargs):
        """
        Add a number of states to the current ODE or to the chosed component
        """
        
        # If comp string is passed we get the component
        if args and isinstance(args[0], str):
            comp = ode
            args = deque(args)
            
            while args and isinstance(args[0], str):
                comp = comp(args.popleft())
        else:
            comp = ode.present_component
            
        # Add the states
        comp.add_states(*args, **kwargs)
    
    def parameters(*args, **kwargs):
        """
        Add a number of parameters to the current ODE or to the chosed component
        """
        
        # If comp string is passed we get the component
        if args and isinstance(args[0], str):
            comp = ode
            args = deque(args)
            
            while args and isinstance(args[0], str):
                comp = comp(args.popleft())
        else:
            comp = ode.present_component
            
        # Add the parameters and update namespace
        comp.add_parameters(*args, **kwargs)
        
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
        ns = {}
        for key, value in kwargs.items():
            if key not in load_arguments:
                ns[key] = value
            else:
                ns[key] = load_arguments[key]
        
        namespace.update(ns)

    def component(*args):
        """
        Set the present component
        """
        
        check_arg(args, tuple, 0, itemtypes=str)
        
        comp = ode
        args = deque(args)
            
        while args:
            comp = comp(args.popleft())

        assert ode.present_component == comp
        return comp

    def markov_model(*args):
        """        
        Initalize a Markov model

        Arguments
        ---------
        args : tuple of str
            Name of components and the Markov model
        """

        check_arg(args, tuple, 0, itemtypes=str)
        
        comp = ode
        args = deque(args)
            
        while len(args)>1:
            comp = comp(args.popleft())

        comp = com.add_markov_model(args[0])

        assert ode.present_component == comp

    # Update provided namespace
    namespace.update(dict(
        states=states,
        parameters=parameters,
        model_arguments=model_arguments,
        component=component,
        subode=subode,
        markov_model=markov_model,
        )
                     )
                    