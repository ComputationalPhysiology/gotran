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
import modelparameters.sympytools as sp_tools
from modelparameters.units import Unit

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
        return [p for p in lst if string.lower() in p.name.lower()]

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
        
        return object.__new__(cls)
        
        
        
        
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
        global _current_cellmodel
        _current_cellmodel = self

        
        # Perhaps more here later
        
        

    @property
    def parameter_symbols(self):
        return [s.name for s in self.parameters]

    @property
    def component_names(self):
        return [s.name for s in self.components]

    def parameter_values(self):
        return [s.value for s in self.parameters]
    
    @property
    def state_symbols(self):
        return [s.name for s in self.states]

    def state_values(self):
        return [s.value for s in self.states]


    def intermediate_unit(self, name, unit_type="si", return_factor=False):
        """
        Get unit of intermediate expression
        Note that we neglect units within a funtion like
        exponential and logaritm.

        Arguments
        ---------
        name : str
            Name of intermediate
        unit_type : str
            Type of unit, options `si`, `base` or `original`

        Returns
        -------
        unit : str
            Unit of the expression for the intermediate

        """
        
        
        def get_intermediate_unit(name, unit_type):
            check_arginlist(name, self.intermediate_symbols)
            intermediate = self.intermediates[self.intermediate_symbols.index(name)]

            factor = 1.0
            # Extract expression
            expr = intermediate.expr.copy()
          
            if isinstance(expr, sp.Piecewise):
                expr = expr.args[0][0]

            
            expr = expr.replace(sp.log, lambda t : 1)
            expr = expr.replace(sp.exp, lambda t : 1)
            expr = expr.replace(sp.zoo, lambda : 1)
            
            unit_dep_map = {}
            
            
            for dep in sp_tools.symbols_from_expr(expr):

                
                dep_str = str(dep).rsplit("(")[0]
                              
                if dep_str in self.parameter_symbols:
                    p = self.get_parameter(str(dep))
                    unit = p.unit


                elif dep_str in self.state_symbols:
                    p = self.get_state(dep_str)
                    unit = p.unit

                elif dep_str in self.intermediate_symbols:
                    unit, factor_= get_intermediate_unit(dep_str, "original")
                    # factor *= factor_

                else:
                    # Parmaterer not found (most likely the time variable)
                    continue
                    

               
                expr1 = expr.subs(dep, unit)
                if expr1 == 0:
                    # We got some cancellation because of the previous substitutioin

                    # Let us add a dummy number
                    expr = expr.subs(dep, "*".join(["2.0", unit]))
                else:
                    expr = expr1

            
                unit_dep_map[dep]=unit
                       
            # Substitute again 
            for k, v in list(unit_dep_map.items()):
                for k1, v1 in list(unit_dep_map.items()):
                    k = k.subs(k1, v1)
                expr = expr.subs(k, v)
                
            # Fix fractions and remove possible numbers
            unit_exprs = []

           
            def add_unit(k,v):
                if not k.is_Number and v.as_numer_denom()[1] == 1: 

                    exp = "**{}".format(str(v)) if v != 1 else ""
                    unit_ = str(k)
                    
                    unit_exprs.append(unit_+exp)
                    
            
            for k, v in list(expr.as_powers_dict().items()):

                # If term consist of multiple term, choose one of them
                if k.is_Add:
                    k = k.as_coeff_add()[1][0]
            
                    for k1, v1 in list(k.as_powers_dict().items()):
                        add_unit(k1, v1)
                    
                else:
                    add_unit(k, v)


            # Join by multiplication
            unit_expr = "*".join(unit_exprs)

            # Check if this is a sum and use only one term
            unit_expr = unit_expr.split(" + ")[0].split(" - ")[0]
     
            # Strip away any numbers but collect the factor
            subunits = "^".join(unit_expr.split("**")).split("*")
            new_subunits = []

            def isfloat(el):
                try: float(el)
                except: return False
                else: return True

    
            for u in subunits:
                if not isfloat(u):
                    new_subunits.append(u)
                

            # Join new expression
            unit_expr = "**".join("*".join(new_subunits).split("^"))

            if unit_expr == "": unit_expr = "1"

            unit = Unit(unit_expr)

            if unit_type == "si":
                factor = unit.si_unit_factor
                retunit = unit.si_unit
            elif unit_type == "base":
                factor = unit.factor
                retunit = unit.base_unit
            elif unit_type == "original":
                factor = 1.0
                retunit = unit.unit

            return retunit, factor
            
        unit_, factor_ = get_intermediate_unit(name, unit_type)

        if return_factor:
            return unit_, factor_
        return unit_


    def set_residual_current(self, t, current):
        """
        Set rediual current

        Arguments
        ---------

        t : array
            List of times
        current : array
            List with residual current
        
        """
        from scipy.interpolate import UnivariateSpline
        self._residual_current = UnivariateSpline(t, current, s= 0)

    def residual_current(self, t):

        if not hasattr(self, "_residual_current"):
            return np.zeros_like(t)

        return self._residual_current(t)
        
       

    @property
    def intermediate_symbols(self):
        return [i.name for i in self.intermediates]
        
    @property
    def stimulation_parameters(self):
        return get_parameter_list_from_string("stim", self.parameters)

    @property
    def stimulation_protocol(self):

        stim_params = self.stimulation_parameters

        if stim_params:
            amplitude =  get_parameter_list_from_string("amp", stim_params)[0]
            duration = get_parameter_list_from_string("dur", stim_params)[0]
            
            period = get_parameter_list_from_string("period", stim_params)
            frequency = get_parameter_list_from_string("frequency", stim_params)
            if period: period = period[0]    
            if frequency: frequency=frequency[0]
            
            start = get_parameter_list_from_string("start", stim_params)[0]
            end = get_parameter_list_from_string("end", stim_params)[0]
        else:
            amplitude = Parameter("amplidude", 0)
            duration = Parameter("duration", 0)
            period = Parameter("period", 500)
            start = Parameter("start", 0)
            end = Parameter("end", 1000)
            frequency=None



        if not period and frequency:
            # Let the period be the reciprocal of the frequency
            unit = Unit(frequency.unit)
            period = Parameter("period", ScalarParam(60/frequency.value,
                                                     unit=unit.reciprocal))
        if not frequency and period:
            unit = Unit(period.unit)
            frequency = Parameter("frequency", ScalarParam(60/period.value,
                                                           unit=unit.reciprocal))

        class StimDict(dict):
            def set(self, key, value):
                self[key].update(value)
                if key == "period":
                    self["frequency"].update(60.0/value)
                elif key == "frequency":
                    self["period"].update(60.0/value)
            
            
        return StimDict(amplitude=amplitude,
                        duration=duration,
                        period=period,
                        frequency=frequency,
                        start=start,
                        end=end)

    @property
    def currents(self):
        """
        Return a list of the currents used in the 
        computation of the membrane potential.
        Note that intermediate currents (not involved in
        th computation of the membrane potential) 
        are not retured
        """

        dV_dt = get_parameter_list_from_string("dV_dt", self.state_expressions)[0]
        currents = [str(type(c)) for c in dV_dt.dependent]
        return currents

    def get_beattime(self, dt=1.0, extra=60.0):
        """
        Return number timepoints for one beat

        Arguments
        ---------
        dt : float
            Time increment in the same time unit 
            as the model
        extra : float
            Add some extra time (in ms) to prolonge the
            the time. Default 60 ms
            
        """

        # Get stimulation parameters
        stim_params = self.stimulation_protocol
    
        # Get duration of one beat
        if stim_params["period"]:
            period = stim_params["period"].value
        else:
            period = 60.0/stim_params["frequency"].value
            
        # Get duration of simulus
        duration = stim_params["duration"].value

        
        factor = 1e-3 if stim_params["duration"].param.unit == "s" else 1.0
        # Include additional 60 ms before stimulation
        extra_ = factor * extra
        beattime = int((period+duration+extra_) / float(dt)) + 1
        
        return beattime

    def get_time_steps(self, nbeats=1, t1=None, dt=1.0, t0 = 0.0):
        """
        Get list with time steps given the number 
        of beats and time increment

        Arguments
        ---------
        nbeats : int
            Nuber of beats (Default:1)
        dt : float
            Time increament between two time steps. (Default:1.0)
            Note that you need to think about the time unit. 
            If time unit is `ms` then dt = 1.0 is probably OK, but
            if time unit is `s` then dt should probably be lower
        t1 : float
            End time. If not provided then end time will determined from
            the number of beats
        t0 : float
            Start time (Default: 0.0)

        """

        # Get stimulation prototocal
        stim_params = self.stimulation_protocol
        
        
        # We let the period of the frequency
        # define the lenght of each beat
        if stim_params["period"]:
            period = stim_params["period"].value
        else:
            period = 60.0/stim_params["frequency"].value
               
        if t1 is None:
            # Use the stimultation protocol to determine the end time
            t1 = t0 + nbeats * period

        # Estimate number of steps
        nsteps = int(t1/float(dt))+1
        tsteps = np.linspace(t0, t1, nsteps)
        return tsteps
        
                

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

        # Parameters for time
        t_end = kwargs.pop("t_end", None)
        dt = kwargs.pop("dt", 0.1)

        # Number of beats
        nbeats = kwargs.pop("nbeats", 1)

        # Number of points (only assimulo)
        npts = kwargs.pop("npts", None)

        # Solver backend
        backend = kwargs.pop("backend", "goss")
        # Solver meothd
        method = kwargs.pop("solver_method", "RKF32")
        # Residual current
        update_residual_current = kwargs.pop("residual_current", False)        
        # Return only the final beat (if simulating multiple beats)
        return_final_beat = kwargs.pop("return_final_beat", False)
        # Return monitored values
        return_monitored = kwargs.pop("return_monitored", False)

        # Get stimulation prototocal
        stim_params = self.stimulation_protocol

        # We let the period of the frequency
        # define the lenght of each beat
        if stim_params["period"]:
            period = stim_params["period"].value
        else:
            period = 60.0/stim_params["frequency"].value
               
        if t_end is None:
            # Use the stimultation protocol to determine the end time
            t0 =  stim_params["start"]
            t_end = t0 + nbeats * period

        # Estimate number of steps
        nsteps = int(t_end/float(dt))

        # Find start and end indices
        if return_final_beat:
            start_idx = int(nsteps*( 1 - (period + \
                                          stim_params["start"].value) \
                                     / float(t_end)) -1)
            end_idx = int(nsteps)
        else:
            start_idx = 0
            end_idx = nsteps


        # Set the end value for stimulation
        stim_params["end"].param.setvalue(t_end, False)

        if backend == "goss":

            import goss
            

            msg = ("Solver method has to be one of "+
                   "{}, got {}".format(goss.goss_solvers, method))
            assert method in goss.goss_solvers, msg

            monitored_symbols = self.intermediate_symbols if return_monitored else None

            module = goss.jit(self, monitored=monitored_symbols)

            solver = getattr(goss, method)(module)
            x0 = module.init_state_values()
            t = 0.0
            
            ys = np.zeros((nsteps, module.num_states()))
            if return_monitored:
                monitored = np.zeros((nsteps, module.num_monitored()))
                monitor = np.zeros(module.num_monitored())
            ts = np.zeros(nsteps)
          
            for step in range(nsteps):

                if update_residual_current:
                    
                    module.set_parameter("i_res_amp", float(self.residual_current(t)))
                    # from IPython import embed; embed()
                    # exit()
                    # print t
                    # solver = getattr(goss, method)(module)
                    # print float(self.residual_current(t))

                if return_monitored:
                    module.eval_monitored(x0, t, monitor)
                    monitored[step,:] = monitor
             
                

                # print t
                solver.forward(x0, t, dt)
             
                ys[step,:] = x0
                ts[step] = t
                t += dt

           
            ret = [ts, ys]

            if return_monitored:
                ret.append(monitored)
           

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
            exec(code, module.__dict__)

            ode = module.ODESim()
            from cellmodels.odesolver import ODESolver

            states= ode.init_state_values()


            options = ODESolver.list_solver_options(method)
            atol = kwargs.pop("atol", None)

            if atol: options["atol"] = atol * np.ones(len(states))

            for k, v in kwargs.items():
                if v and k in options: options[k] = v


            solver = ODESolver(ode, states, method = method, **options)

            t_, y_ = solver.solve(t_end, nbeats)

            ret = [t_, y_]

        if return_final_beat:

            ret[0] = ret[0][start_idx:end_idx]
            ret[1] = ret[1][start_idx:end_idx]
            
        return ret
    
        
    def get_component(self, name):
        """
        Get the component with the given name

        Arguments
        ---------

        name : str
            Name of the component

        Returns
        -------
        par : ODECompoenent
            The component

        """

        # Check if parameter exist
        check_arginlist(name, self.component_names)
        # Get the index
        idx = self.component_names.index(name)
        # Return the component
        return self.components[idx]
        
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

    def get_state(self, name):
        """
        Get the state with the given name

        Arguments
        ---------

        name : str
            Name of the state variable

        Returns
        -------
        par : Parameter
            The state

        """
        # Check if parameter exist
        check_arginlist(name, self.state_symbols)
        # Get the index
        idx = self.state_symbols.index(name)
        # Return the parameter
        return self.states[idx]

    def get_intermediate(self, name):
        """
        Get the intermediate with the given name

        Arguments
        ---------

        name : str
            Name of the intermediate

        Returns
        -------
        par : Parameter
            The parameter

        """
        # Check if parameter exist
        check_arginlist(name, self.intermediate_symbols)
        # Get the index
        idx = self.intermediate_symbols.index(name)
        # Return the parameter
        return self.intermediates[idx]


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
    
