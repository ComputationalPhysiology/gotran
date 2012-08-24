from modelparameters.parameterdict import *

from gotran2.model.ode import ODE
from gotran2.common import check_arg

def _default_params():
    return ParameterDict(
        field_states = False,
        field_parameters = False,
        use_variables = False,
        parameter_contraction = False,
        parameter_numerals = False,
        max_terms = ScalarParam(10, ge=2),
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
