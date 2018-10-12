# Global imports
import signal
import numpy as np






gotran_methods = ['explicit_euler',
                 'rush_larsen', 'generalized_rush_larsen',
                 'simplified_implicit_euler']

sundials_methods = ["cvode"]

goss_methods = ["RKF32"]

methods = ["scipy"] + gotran_methods + sundials_methods + goss_methods

class ODESolverError(Exception):pass


def timeout(timeout_seconds):
    def decorate(function):
        message = "Timeout (%s sec) elapsed for solve %s" % (timeout_seconds, function.__name__)



        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)

            try:
                function_result = function(*args, **kwargs)
            finally:
                signal.signal(signal.SIGALRM, old)
            signal.alarm(0)
            return function_result

        new_f.func_name = function.func_name
        return new_f

    return decorate

class Solver(object):
    def __init__(self, ode, **options):

        # Set 1000 seconds as max
        self.max_solve_time = options.get('max_solve_time', 1000)

        # Get names of monitoed expression
        self._monitored = []
        for expr in sorted(ode.intermediates + ode.state_expressions):
            self._monitored.append(expr.name)

        # Get names of states
        self._state_names = ode.state_symbols

        # Create C module
        self._module = generate_module(ode, self._monitored, **options)

        # The right hand side
        self._rhs = self._module.rhs
        # The jacobian
        self._jac = None if not hasattr(self.module,'compute_jacobian') \
                    else self.module.compute_jacobian

        # Initial conditions for the states
        self._y0 = self.module.init_state_values()

        # The model parameters
        self._model_params = self.module.init_parameter_values()

        self._ode = ode


    def update_model_parameter(self):
        """
        Update model parameters according
        to parameters in the cell model
        """
        self._model_params = np.array(self._ode.parameter_values(), dtype='float64')

    def solve(self, *args, **kwargs):
        """
        Solver ODE. See docs for the different solvers
        for solver specific arguments.
        """

        self.update_model_parameter()

        # def handler(signum, frame):
        #     raise TimeoutError(('Time to solve ODE has exceeded '
        #                         'the max solving time'))
        # old = signal.signal(signal.SIGALRM, handler)
        # signal.alarm(self.max_solve_time)
        # try:
        #     print('Solving with max solve time = {}'.format(self.max_solve_time))
        ret = self._solve(*args, **kwargs)
        # finally:
        #     signal.signal(signal.SIGALRM, old)
        # signal.alarm(0)
        return ret

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

            self._eval_monitored(time, res, self._model_params,
                                monitored_get_values)
            monitored_results[ind,:] = monitored_get_values


        return monitored_results

    def _eval_monitored(self, time, res, params, values):
        self.module.monitor(res, time,  params, values)



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


    generation.code.default_arguments \
        = options.pop("arguments", "tsp")

    generation.functions.rhs.generate = True
    generation.functions.jacobian.generate \
        = options.pop("generate_jacobian", False)

    # Language for the module ("C" or "Python")
    language = options.pop("language", "C")

    additional_declarations \
        = options.pop("additional_declarations", None)
    jacobian_declaration_template \
        =options.pop("jacobian_declaration_template", None)



    module = compile_module(ode, language,monitored,generation,
                            additional_declarations,
                            jacobian_declaration_template)

    return module



# Local imports
from .sundialssolver import SundialsSolver, SundialsNotInstalled
from .scipysolver import ScipySolver


def ODESolver(ode, method="scipy", **options):
     """
    A generic ODE solver for solving problem of the types on the form,

    .. math::

        \dot{y} = f(t,y), \quad y(t_0) = y_0.

    Here one need to specific the backend which is either Scipy or Assimulo.

    *Arguments*

    ode : gotran.ODE or gotran.CellModel
        The ode you want to solve in a gotran object
    method : str
       Solver method. Possible inputs are  or 'scipy' (Default:'sundials')

    options : dict:
       Options for the solver, see `list_solver_options`
    """

     check_method(method)

     if method == "scipy":
         return ScipySolver(ode, **options)
     elif method in sundials_methods:
          try:
               return SundialsSolver(ode, method, **options)
          except:
               print("Could not import Sundials solvers. Use Scipy ODE solver instead")
               return ScipySolver(ode)

     elif method in gotran_methods:
         raise NotImplementedError
     elif method in goss_methods:
         raise NotImplementedError
     else:
         raise NotImplementedError
