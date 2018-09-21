import warnings
from copy import deepcopy
try:
    import scipy.integrate as spi
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
        self._options.update( (k,v) for k,v in options.items() \
                              if k in list(self._options.keys()))


    def get_options(self):
        return self._options

    @staticmethod
    def list_solver_options():
        return {'atol': None,
                'col_deriv': 0,
                'full_output': 0,
                'h0': 0.0,
                'hmax': 1e-3,
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
        
    def _solve(self, tsteps, attempts = 3):
        """
        Solve ode using scipy.integrade.odeint

        Arguments
        ---------
        tsteps : array
            The time steps
        attempts : int
            If integration fails, i.e the solver does not 
            converge, we could reduce the step size and try again.
            This varible controls how many time we should try to
            solve the problem. Default: 3
        
        """

        
        # Some flags
        it = 0
        converged = False

        # Get solver options
        options = deepcopy(self._options)
      
        while it < attempts and not converged:

            
            # Somehow scipy only display a warning if the ODE itegrator fails.
            # We can record these warnings using the warning module
            with warnings.catch_warnings(record=True) as caught_warnings:

                # Allways catch warnings (not only the first)
                warnings.simplefilter("always")
                
                # Solve ode
                results = spi.odeint(self._rhs, self._y0,
                                     tsteps, Dfun=self._jac,
                                     args=(self._model_params,),
                                     **options)

            # Check if we caught any warnings 
            converged = len(caught_warnings) == 0
            it += 1
            # If we did, reduce maximum step size
            options["hmax"] /= 2.0

        # If we still caught some warnings raise exception
        if len(caught_warnings) > 0:
            for w in caught_warnings:
                msg="Catched warning {}\n{}".format(w.category,
                                                    w.message)
                warning(msg)

                if w.category == spi.odepack.ODEintWarning:
                    raise ODESolverError(msg)
            
            
        return tsteps, results
