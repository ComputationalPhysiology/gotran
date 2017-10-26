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

# FIXME: This file is depricated...

__all__ = ["CellModel", "gccm"]

from gotran.common import *
from gotran.model.ode import ODE
from gotran.model.odeobjects import *
# from gotran.common import error, debug, check_arg, check_kwarg, check_arginlist

from modelparameters.parameters import *

# Holder for all cellmodels
_all_cellmodels = {}

# Holder for current CellModel
_current_cellmodel = None

def get_parameter_list_from_string(string, lst, case_insesitive=True):
    """
    Return a list with parameters in the given list
    that has a name containing the given string

    Arguments
    ---------

    string : str
        The string that should be in the name
    lst : array with parameters
        List containing the candidates for parameters
    case_insesitive: bool
        If True, include parameter as long as the letters coincide
        and do not care about case

    Returns
    -------
    parlist : lst
        A sublist of lst containing the parameter that has a name
        containing the given string
    
    """
    if case_insesitive:
        return [p for p in lst if string in p.name.lower()]

    else:
        return [p for p in lst if string in p.name]
    

class CellModel(ODE):
    """
    Basic class for storing information of a cell model

    You can either initialize an empty CellModel similar to
    an ODE

    .. Example::

        cell =  CellModel("MyCell")

    (Maybe if we can create a library of th cellmodels
    then it the string matched that of the library we return
    the model in the library, e.g cell = CellModel("Noble_1962") ?? )
    Or you can initialize the cell using an existing ODE

    If you have an ODE and want to have a cell, then save the 
    ODE to an *.ode file and load it using 'load_cell' (in stead of
    'load_ode')

    .. Example::

        # Save current ODE
        ode.save(filename)

        # Load cell
        from loadmodel import load_cell
        cell = load_cell(filename)

    """
    def __new__(cls, *args, **kwargs):
        """
        Create a CellModel instance.
        """

        arg = args[0]
        if arg in _all_cellmodels:
            return _all_cellmodels[arg]
        
        return object.__new__(cls, *args, **kwargs)
        
        
        
        
    def __init__(self, name_, ns=None):
        """
        Initialize a CellModel

        """

        check_arg(name_, str)
        name = name_
        
        super(CellModel, self).__init__(name, ns)
        

        # Do not reinitialized object if it already excists
        if name in _all_cellmodels:
            return

        # Initialize attributes
        self.name = name
        
        self._initialize_cell_model()
        
        # Store instance for future lookups
        _all_cellmodels[name] = self
        
        

    def _initialize_cell_model(self):
        # Set current CellModel
        _current_cellmodel = self


        # Perhaps out more here later
        

    @property
    def parameter_symbols(self):
        return [s.name for s in self.parameters]

    @property
    def component_names(self):
        return [s.name for s in self.components]

    def parameter_values(self):
        return [s.value for s in self.parameter]
    
    @property
    def state_symbols(self):
        return [s.name for s in self.states]

    def state_values(self):
        return [s.value for s in self.states]

    @property
    def stimulation_parameters(self):
        return get_parameter_list_from_string("stim", self.parameters)

    @property
    def stimulation_protocol(self):

        stim_params = self.stimulation_parameters
 
        if stim_params:
            amplitude =  get_parameter_list_from_string("amp", stim_params)[0]
            duration = get_parameter_list_from_string("dur", stim_params)[0]
            period = get_parameter_list_from_string("period", stim_params)[0]
            start = get_parameter_list_from_string("start", stim_params)[0]
            end = get_parameter_list_from_string("end", stim_params)[0]
        else:
            amplitude = Parameter("amplidude", 0)
            duration = Parameter("duractioN", 0)
            period = Parameter("period", 500)
            start = Parameter("start", 0)
            end = Parameter("end", 1000)
            
        return {"amplitude":amplitude,
                "duration": duration,
                "period": period,
                "start": start,
                "end": end}
                

    def simulate(self, **kwargs):
        """
        Simulate the ODE to :math:`t_{\mathrm{end}}`
        with the given number points 

        Aguments
        --------

        t_end : scalar
            The end time
        nbeats : scalar
            Number of beats based on stimulus protocol
        npts : int
            Number of communication points used in the solver
        
        """


        t_end = kwargs.pop("t_end", None)
        dt = kwargs.pop("dt", 0.1)
        nbeats = kwargs.pop("nbeats", 1)
        npts = kwargs.pop("npts", None)
        
        backend = kwargs.pop("backend", "goss")
        method = kwargs.pop("solver_method", "RKF32")
        return_final_beat = kwargs.pop("return_final_beat", False)
       
        stim_params = self.stimulation_protocol
               
        if t_end is None:
            # Use the stimultation protocol to determine the end time
            t0 =  stim_params["start"]
            t_end = t0 + nbeats * stim_params["period"]

        nsteps = int(t_end/float(dt))
        
        if return_final_beat:

            start_idx = int(nsteps*( 1 - (stim_params["period"].value \
                                          + stim_params["start"].value) \
                                     / float(t_end)) -1)
            end_idx = int(nsteps)
        else:
            start_idx = 0
            end_idx = nsteps

        stim_params["end"].value = t_end

        if backend == "goss":

            import goss
            

            msg = ("Solver method has to be one of "+
                   "{}, got {}".format(goss.goss_solvers, method))
            assert method in goss.goss_solvers, msg

            module = goss.jit(self)

            solver = getattr(goss, method)(module)
            x0 = module.init_state_values()
            t = 0.0

            
            ys = np.zeros((nsteps, len(x0)))
            ts = np.zeros(nsteps)
          
            for step in range(nsteps):
          
                solver.forward(x0, t, dt)
             
                ys[step,:] = x0
                ts[step] = t
                t += dt

           
            ret = [ts, ys]
           

        else:
            from gotran.codegeneration.codegenerators import PythonCodeGenerator
            from gotran.common.options import parameters
            import imp

    
            params = parameters.generation.copy()
            params.functions.rhs.function_name="__call__"
            params.code.default_arguments="tsp" 
            params.class_code=1
   

            monitored = [expr.name for expr in self.intermediates + self.state_expressions]
            gen = PythonCodeGenerator(params)

            name = self.name
            self.rename("ODESim")
            code = gen.class_code(self, monitored)
            self.rename(name)


            module = imp.new_module("simulation")
            exec code in module.__dict__

            ode = module.ODESim()
            from cellmodels.odesolver import ODESolver

            states= ode.init_state_values()


            options = ODESolver.list_solver_options(method)
            atol = kwargs.pop("atol", None)

            if atol: options["atol"] = atol * np.ones(len(states))

            for k, v in kwargs.iteritems():
                if v and k in options: options[k] = v


            solver = ODESolver(ode, states, method = method, **options)

            t_, y_ = solver.solve(t_end, nbeats)

            ret = [t_, y_]

        if return_final_beat:

            ret[0] = ret[0][start_idx:end_idx]
            ret[1] = ret[1][start_idx:end_idx]
            
        return ret
    
        
        
        
    def get_parameter(self, name):
        """
        Get the parameter with the given name

        Arguments
        ---------

        name : str
            Name of the parameter

        Returns
        -------
        par : Parameter
            The parameter

        """
        # Check if parameter exist
        check_arginlist(name, self.parameter_symbols)
        # Get the index
        idx = self.parameter_symbols.index(name)
        # Return the parameter
        return self.parameters[idx]

    def get_parameter(self, name):
        """
        Get the parameter with the given name

        Arguments
        ---------

        name : str
            Name of the parameter

        Returns
        -------
        par : Parameter
            The parameter

        """
        # Check if parameter exist
        check_arginlist(name, self.parameter_symbols)
        # Get the index
        idx = self.parameter_symbols.index(name)
        # Return the parameter
        return self.parameters[idx]

    def set_parameter(self, name, value):
        """
        Set the parameter in the model to a
        given value


        Arguments
        ---------

        name : str
            Name of the parmaeter
        value : scalar, ScalarParam
            The new value of the parameter. Note that
            if the parameter is of type `ScalarParam` 
            while the provided value is a scalar then the value
            will be updated while keeping the unit

        """
        check_arg(value, scalars + (ScalarParam,), 1, Parameter)
        
        par = self.get_parameter(name)
        if isinstance(value, ScalarParam):
            par._param = value
        else:
            par.value = value

    def update_parameter(self, name, value=None, factor=1.0):

        if value is None:
            par = self.get_parameter(name)
            par._param.value *= factor
        else:
            self.set_parameter(name, value)

# Construct a default CellModel
# _current_cellmodel = CellModel("Default")
        
def gccm():
    """
    Return the current CellModel
    """
    assert(isinstance(_current_cellmodel, CellModel))
    return _current_cellmodel
    
