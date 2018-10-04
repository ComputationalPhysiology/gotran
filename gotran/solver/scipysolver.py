import warnings
from copy import deepcopy
import numpy as np
try:
    import scipy.integrate as spi
    has_scipy = True
except:
    has_scipy = False

from gotran.common import warning
from .odesolver import Solver, ODESolverError

__all__ = ["ScipySolver", "has_scipy"]


class ScipySolver(Solver):
    def __init__(self, ode, **options):

        msg = "Chosen backend is scipy, but scipy is not installed"
        assert has_scipy, msg

        Solver.__init__(self, ode, **options)

        self._options =  ScipySolver.list_solver_options()
        self._options.update((k,v) for k,v in options.items() \
                              if k in list(self._options.keys()))

    def get_options(self):
        return self._options

    # @staticmethod
    # def list_solver_options():
    #     return {'atol': 1e-6,
    #             'max_step': np.inf,
    #             'rtol': 1e-6}

    # These are the old ones
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
        
    def _solve(self, tsteps, attempts=3):
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

                fun=lambda y, t: self._rhs(t, y, self._model_params)
                # Solve ode
                results = spi.odeint(fun, self._y0,
                                     tsteps, Dfun=self._jac,
                                     **options)
                t, y = tsteps, results
                
                
                # fun=lambda t, y: self._rhs(t, y, self._model_params)
                # results = spi.solve_ivp(fun=fun,
                #                         y0=self._y0,
                #                         t_span=[tsteps[0], tsteps[-1]],
                #                         method='BDF', #'BDF', 'LSODA' 'Radau'
                #                         t_eval=tsteps,
                #                         jac=self._jac,
                #                         **options)
                # t, y = results.t, results.y

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

        return t, y
