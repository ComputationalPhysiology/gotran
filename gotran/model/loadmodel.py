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

__all__ = ["load_ode", "exec_ode", "load_cell", "get_model_as_python_module"]

# System imports
import inspect
import os
import shutil
import re
import tempfile
import weakref
from collections import OrderedDict, deque

# modelparameters import
from modelparameters.parameters import Param, ScalarParam, ArrayParam, \
     ConstParam, scalars, OptionParam
from modelparameters.sympytools import sp_namespace, sp

# gotran imports
from gotran.common import *
from gotran.model.ode import ODE
from gotran.model.cellmodel import CellModel

_for_template = re.compile("\A[ ]*for[ ]+.*in[ ]+:[ ]*\Z")
_no_intermediate_template = re.compile(".*# NO INTERMEDIATE.*")

def get_model_as_python_module(model):
    from gotran.common.options import parameters
    from gotran.codegeneration.codegenerators import PythonCodeGenerator
    

    import imp

    
    params = parameters.generation.copy()
    params.functions.rhs.function_name="__call__"
    params.code.default_arguments="tsp" 
    params.class_code=1
    

    monitored = [expr.name for expr in model.intermediates + model.state_expressions]
    gen = PythonCodeGenerator(params)
    
    name = model.name
    model.rename("ODESim")
    code = gen.class_code(model, monitored)
    model.rename(name)

    module = imp.new_module("simulation")
    exec(code, module.__dict__)
    return module.ODESim()

class IntermediateDispatcher(dict):
    """
    Dispatch intermediates to ODE attributes
    """
    
    def __init__(self, ode=None):
        """
        Initalize with an ode
        """
        if ode is not None:
            self._ode = weakref.ref(ode)
        else:
            self._ode = None


    @property
    def ode(self):

        if self._ode == None:
            error("ode attr is not set")
        
        return self._ode()

    @ode.setter
    def ode(self, ode):
        self._ode = weakref.ref(ode)

    def __setitem__(self, name, value):
        """
        This is only for expressions
        """

        from modelparameters.sympytools import symbols_from_expr
        timer = Timer("Namespace dispatcher")
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

                del timer
                
                # Add obj to the present component
                sym = setattr(self.ode.present_component, name, value)
        
        else:
            debug("Not registering '{0}' as an intermediate.".format(name))

            # If no ode attr was generated we just add the value to the
            # namespace
            dict.__setitem__(self, name, value)

    def update(self, other):
        check_arg(other, dict)
        for name, value in list(other.items()):
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
                          OptionParam=OptionParam,
                          sp=sp,
                          ))

    # Add ode and model_arguments
    _namespace_binder(namespace, weakref.ref(ode), load_arguments)

    return namespace

def exec_ode(ode_str, name, **arguments):
    """
    Execute an ode given by a str

    Arguments
    ---------
    ode_str : str
        The ode as a str
    name : str 
        The name of ODE
    arguments : dict (optional)
        Optional arguments which can control loading of model
    """
    # Dict to collect declared intermediates
    intermediate_dispatcher = IntermediateDispatcher()

    # Create an ODE which will be populated with data when ode file is loaded
    ode = ODE(name, intermediate_dispatcher)

    intermediate_dispatcher.ode = ode

    debug("Loading {}".format(ode.name))

    # Create namespace which the ode file will be executed in
    _init_namespace(ode, arguments, intermediate_dispatcher)

    # Write str to file
    with open("_tmp_gotrand.ode", "w") as f:
        f.write(ode_str)
    
    # Execute file and collect
    with open("_tmp_gotrand.ode") as f:
        exec(compile(f.read(), "_tmp_gotrand.ode", 'exec'),
             intermediate_dispatcher)
    os.unlink("_tmp_gotrand.ode")

    # Finalize ODE
    ode.finalize()
    
    # Check for completeness
    if not ode.is_complete:
        warning("ODE mode '{0}' is not complete.".format(ode.name))
    
    info("Loaded ODE model '{0}' with:".format(ode.name))
    for what in ["full_states", "parameters"]:
        num = getattr(ode, "num_{0}".format(what))
        info("{0}: {1}".format(("Num "+what.replace("_", \
                                                    " ")).rjust(20), num))
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
    
    arguments["class_type"] = "ode"
    return _load(filename, name, **arguments)

def _load(filename, name, **arguments):
    """
    Load an ODE or Cell from file and return the instance

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
    # Extract name from filename
    if len(filename) < 4 or filename[-4:] != ".ode":
        name = name or filename
        filename = filename + ".ode"
    elif name is None:
        name = filename[:-4]


    # Execute the file
    if (not os.path.isfile(filename)):
        error("Could not find '{0}'".format(filename))

   
    # Copy file temporary to current directory
    basename = os.path.basename(filename)
    copyfile = False
    if not basename == filename:
        shutil.copy(filename, basename)
        filename = basename
        name = filename[:-4]
        copyfile=True
    
    
    # If a Param is provided turn it into its value
    for key, value in list(arguments.items()):
        if isinstance(value, Param):
            arguments[key] = value.getvalue()


    class_type = arguments.pop("class_type", "ode")
    msg = "Argument class_type must be one of "\
          "('ode', 'cell'), got %s " % class_type
    assert(class_type in ("ode", "cell")), msg


    # Dict to collect namespace
    intermediate_dispatcher = IntermediateDispatcher()
    

    # Create an ODE which will be populated with data when ode file is loaded
    
    if class_type == "ode":
        ode = ODE(name, intermediate_dispatcher)
    else:
        ode = CellModel(name, intermediate_dispatcher)

    intermediate_dispatcher.ode = ode

    debug("Loading {}".format(ode.name))

    
    # Create namespace which the ode file will be executed in
    _init_namespace(ode, arguments, intermediate_dispatcher)

    # Execute file and collect
    with open(filename, 'r') as f:
        exec(compile(f.read(), filename, 'exec'),
             intermediate_dispatcher)

    # Finalize ODE
    ode.finalize()
    
    # Check for completeness
    if not ode.is_complete:
        warning("ODE model '{0}' is not complete.".format(ode.name))
    
    info("Loaded ODE model '{0}' with:".format(ode.name))
    for what in ["full_states", "parameters"]:
        num = getattr(ode, "num_{0}".format(what))
        info("{0}: {1}".format(("Num "+what.replace("_", \
                                                    " ")).rjust(20), num))
    if copyfile: os.unlink(filename)
    
    return ode



def load_cell(filename, name=None, **arguments):
    """
    Load a Cell from file and return the instance

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
    arguments["class_type"] = "cell"
    
    cell =_load(filename, name, **arguments)
    cell._initialize_cell_model()
    return cell
    

def _namespace_binder(namespace, ode, load_arguments):
    """
    Add functions all bound to current ode, namespace and arguments
    """
    
    
    def comment(comment):
        """
        Add a comment to the present ODE component

        Arguments
        ---------
        comment : str
            The comment
        """
        
        comp = ode().present_component
            
        # Add the comment
        comp.add_comment(comment)
        
    def subode(subode, prefix="", components=None):
        """
        Load an ODE and add it to the present ODE (deprecated)

        Argument
        --------
        subode : str
            The subode which should be added.
        prefix : str (optional)
            A prefix which all state and parameters are prefixed with.
        components : list, tuple of str (optional)
        A list of components which will be extracted and added to the present
            ODE. If not given the whole ODE will be added.
        """
        warning("Usage of 'sub_ode()' is deprecated. "\
                "Use 'import_ode()' instead.")
        import_model(subode, prefix, components)

    def import_ode(subode, prefix="", components=None, **arguments):
        """
        Import an ODE into the present ode

        Argument
        --------
        subode : str
            The ode which should be imported
        prefix : str (optional)
            A prefix which all state, parameters and intermediates are
            prefixed with.
        components : list, tuple of str (optional)
            A list of components which will either be extracted or excluded
            from the imported ode. If not given the whole ODE will be imported.
        arguments : dict (optional)
            Optional arguments which can control loading of model
        """

        check_arg(subode, str, 0)

        # Add the subode and update namespace
        ode().import_ode(subode, prefix=prefix, components=components, **arguments)

    def timeunit(*args, **kwargs):
        """
        Update timeunit according to timeunit in file
        """
        comp = ode()

        if args and isinstance(args[0], str):
            comp._time.update_unit(args[0])
            comp._dt.update_unit(args[0])
        

    def states(*args, **kwargs):
        """
        Add a number of states to the current component or to the
        chosed component
        """
      
        # If comp string is passed we get the component
        if args and isinstance(args[0], str):
            comp = ode()
            args = deque(args)
            
            while args and isinstance(args[0], str):
                comp = comp(args.popleft())
        else:
            comp = ode().present_component
            
        # Update the rates name so it points to the present components
        # rates dictionary
        namespace["rates"] = comp.rates

        # Add the states
        comp.add_states(*args, **kwargs)

        
    
    def parameters(*args, **kwargs):
        """
        Add a number of parameters to the current ODE or to the chosed component
        """
        # If comp string is passed we get the component
        if args and isinstance(args[0], str):
            comp = ode()
            args = deque(args)
            
            while args and isinstance(args[0], str):
                comp = comp(args.popleft())
        else:
            comp = ode().present_component
            
        # Update the rates name so it points to the present components
        # rates dictionary
        namespace["rates"] = comp.rates

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
          >>> load_ode("model", include_Na=False)
          >>> ...
          
        """
    
        # Check the passed load arguments
        for key in load_arguments:
            if key not in kwargs:
                error("Name '{0}' is not a model_argument.".format(key))
    
        # Update the namespace
        ns = {}
        for key, value in list(kwargs.items()):
            if not isinstance(value, (float, int, str, Param)):
                error("expected only 'float', 'int', 'str' or 'Param', as model_arguments, "\
                      "got: '{}' for '{}'".format(type(value).__name__, key))
                
            if key not in load_arguments:
                ns[key] = value.getvalue() if isinstance(value, Param) else value
                    
            else:
                # Try to cast the passed load_arguments with the orginal type
                if isinstance(value, Param):

                    # Cast value
                    new_value = value.value_type(load_arguments[key])

                    # Try to set new value
                    value.setvalue(new_value)

                    # Assign actual value of Param
                    ns[key] = value.getvalue()
                    
                else:
                    ns[key] = type(value)(load_arguments[key])
        
        namespace.update(ns)

    def component(*args):
        """
        Set the present component, deprecated
        """
        warning("Usage of 'component()' is deprecated. "\
                "Use 'expressions()' instead.")
        return expressions(*args)

    def expressions(*args):
    
        check_arg(args, tuple, 0, itemtypes=str)
        
        comp = ode()

        args = deque(args)
            
        while args:
            comp = comp(args.popleft())

        assert ode().present_component == comp

        # Update the rates name so it points to the present components
        # rates dictionary
        namespace["rates"] = comp.rates

        return comp

    # Update provided namespace
    namespace.update(dict(
        timeunit=timeunit,
        states=states,
        parameters=parameters,
        model_arguments=model_arguments,
        component=component,
        expressions=expressions,
        subode=subode,
        import_ode=import_ode,
        comment=comment
        )
                     )
                    
