
import numpy as np

gotran_methods = ['explicit_euler',
                 'rush_larsen', 'generalized_rush_larsen',
                 'simplified_implicit_euler']

sundials_methods = ["cvode", "lsodar"]

goss_methods = ["RKF32"]

methods = ["scipy"] + gotran_methods + sundials_methods + goss_methods


    

class Solver(object):
    def __init__(self, ode, **options):

        # Add monitored
        self._monitored = []
        for expr in sorted(ode.intermediates + ode.state_expressions):
            self._monitored.append(expr.name)
        

        self._state_names = ode.state_symbols

        
        self._module = generate_module(ode, self._monitored, **options)
        self._y0 = self.module.init_state_values()
        self._model_params = self.module.init_parameter_values()

    @property
    def module(self):
        return self._module

    def monitor_indices(self, *monitored):
        return self.module.monitor_indices(*monitored)
    
    @property
    def monitor_names(self):
        return self._monitored

    def state_indices(self, *states):
        return self.module.state_indices(*states)

    @property
    def state_names(self):
        return self._state_names
    
    def monitor(self, tsteps, results):
        """
        Get monitored values
        """
                

        monitored_results = np.zeros((len(tsteps),
                                    len(self.monitor_names)),
                                    dtype=np.float_)
        monitored_get_values = np.zeros(len(self.monitor_names),
                                        dtype=np.float_)

        for ind, (time, res) in enumerate(zip(tsteps, results)):
        
            self.module.monitor(res, time, self._model_params,
                                monitored_get_values)
            monitored_results[ind,:] = monitored_get_values
                                   
            
        return monitored_results
        

    
def check_method(method):
    msg = "Unknown method {1}, possible method are {0}".format(method, methods)
    assert(method in methods), msg


def generate_module(ode, monitored, **options):
    """
    Generate a module to 
    """
    
    from gotran.codegeneration.compilemodule import compile_module
    from gotran.common.options import parameters
    # Copy of default parameters
    generation = parameters.generation.copy()

    generation.functions.monitored.generate\
        = options.pop("generate_monitored", True)

    
        
    generation.functions.rhs.generate = True
    generation.functions.jacobian.generate \
        = options.pop("generate_jacobian", False)

    # Language for the module ("C" or "Python")
    language = options.pop("language", "C")



    module = compile_module(ode, language,monitored,generation)

    return module
