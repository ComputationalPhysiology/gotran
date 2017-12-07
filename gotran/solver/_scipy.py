try:
    from scipy.integrate import odeint
    has_scipy = True
except:
    has_scipy = False


# Local imports
from .utils import *

__all__ = ["ScipySolver", "has_scipy"]


class ScipySolver(Solver):
    def __init__(self, ode, **options):

        msg = "Chosen backend is scipy, but scipy is not installed"
        assert has_scipy, msg

        Solver.__init__(self, ode, **options)

        self._options =  ScipySolver.list_solver_options()
        self._options.update(**options)


    def get_options(self):
        return self._options

    @staticmethod
    def list_solver_options():
        return {'atol': None,
                'col_deriv': 0,
                'full_output': 0,
                'h0': 0.0,
                'hmax': 0.0,
                'hmin': 0.0,
                'ixpr': 0,
                'ml': None,
                'mu': None,
                'mxhnil': 0,
                'mxordn': 12,
                'mxords': 5,
                'mxstep': 0,
                'printmessg': 0,
                'rtol': None,
                'tcrit': None}

    def solve(self, tsteps):

        results = odeint(self._rhs, self._y0,
                         tsteps, Dfun=self._jac,
                         args=(self._model_params,),
                         **self._options)

        return tsteps, results
