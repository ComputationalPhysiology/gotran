from modelparameters.parameterdict import *

from gotran2.model.ode import ODE
from gotran2.common import check_arg

def _default_params():
    return ParameterDict(

        # Use state, parameters, and variable names in code (compared to
        # array with indices)
        use_names = True,

        # Keep all intermediates
        keep_intermediates = True, 

        # If True, logic for field states are created
        field_states = False,

        # If True, logic for field paramters are created
        field_parameters = False,

        # If True , code for altering variables are created
        use_variables = False,

        # Find sub expressions of only parameters and create a dummy parameter
        parameter_contraction = False,

        # Exchange all parameters with their initial numerical values
        parameter_numerals = False,

        # Split terms with more than max_terms into several evaluations
        max_terms = ScalarParam(5, ge=2),

        # Use sympy common sub expression simplifications
        use_cse = False,
        )

class ODERepresentation(object):
    """
    Intermediate ODE representation where various optimizations
    can be performed.
    """
    def __init__(self, ode, **optimization):
        check_arg(ode, ODE, 0)
        
        self.ode = ode
        self.optimization = _default_params()
        self.optimization.update(optimization)
        
    @property
    def name(self):
        return self.ode.name
