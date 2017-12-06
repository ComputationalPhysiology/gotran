# Assimulo imports
try:
    from assimulo.solvers import CVode, LSODAR, IDA, Radau5ODE
    from assimulo.problem import Explicit_Problem
    has_assimulo = True
except:
    has_assimulo = False

# Local imports
from .utils import *

__all__ = ["AssimuloSolver", "has_assimulo"]

class Jac(object):
    def __init__(self, module):
        self._module = module

    def __call__(self, time, states):
        parameters = self._module.init_parameter_values()
        return self._module.compute_jacobian(states, time, parameters)
    

class RHS(object):
    def __init__(self, module):
        self._module = module

    def __call__(self, time, states):
        parameters = self._module.init_parameter_values()
        return self._module.rhs(states, time, parameters)



class AssimuloSolver(Solver):
    
    def __init__(self, ode, method = "cvode", **options):

        # Check imports
        msg = ("Chosen backend is assimulo, but assimulo is "+
               "not installed")
        assert has_assimulo, msg

        # Check method
        msg = ("Method {} is not a sundials method. ".format(method)+\
               "Possible methods are {}".format(sundials_methods))
        assert method in sundials_methods, msg
        self._method = method

        Solver.__init__(self, ode, **options)
        

        self._rhs = RHS(self.module)
        self._jac = None if not hasattr(self.module,'compute_jacobian') \
                    else Jac(self.module)
        
        
        self._problem = Explicit_Problem(self._rhs, self._y0)

        if self._jac is not None:
            self._problem.jac = self._jac
            options['usejac'] = True
        
        
        if method == "cvode":
            self._solver = CVode(self._problem)
        elif method == "ida":
            self._solver = IDA(self._problem)
        elif method == "radau5ode":
            self._solver = Radau5ODE(self._problem)
        else:
            self._solver =  LSODAR(self._problem)


        solver_parameters = AssimuloSolver.list_solver_options(self._method)
        solver_parameters.update(**options)
        self._solver.options.update(**solver_parameters)

    @property
    def problem(self):
        return self._problem

    @property
    def solver(self):
        return self._solver

    
    def get_options(self):
        """
        Get solver options for the current solver
        """
        return self.solver.options
        
    @staticmethod
    def list_solver_options(method):
    
        if method == "lsodar":
            return _LSODAR.default_options()
        elif method == "cvode":
            return _CVode.default_options()
        elif method == "ida":
            return _IDA.default_options()
        elif method == "radau5ode":
            return _Radau5ODE.default_options()

    def solve(self, t_end, ncp = 0, ncp_list = None):
        """
        Solve the problem
        """
        # Construct problem
        
        return self._solver.simulate(t_end, ncp, ncp_list)
    

class _Radau5ODE:
    @staticmethod
    def default_options():
        d = {'backward': False,
             'clock_step': False,
             'display_progress': True,
             'fac1': 0.2,
             'fac2': 8.0,
             'fnewt': 0.0,
             'inith': 0.01,
             'maxh': 10.0,
             'maxsteps': 100000,
             'newt': 7,
             'num_threads': 1,
             'quot1': 1.0,
             'quot2': 1.2,
             'report_continuously': False,
             'rtol': 1e-06,
             'safe': 0.9,
             'store_event_points': True,
             'thet': 0.001,
             'time_limit': 0,
             'usejac': False,
             'verbosity': 30}
        return d
        
class _IDA:
    @staticmethod
    def default_options():
        d = {'backward': False,
             'clock_step': False,
             'display_progress': True,
             'dqrhomax': 0.0,
             'dqtype': 'CENTERED',
             'external_event_detection': False,
             'inith': 0.0,
             'linear_solver': 'DENSE',
             'lsoff': False,
             'maxcorS': 3,
             'maxh': 0.0,
             'maxord': 5,
             'maxsteps': 10000,
             'num_threads': 1,
             'pbar': [],
             'report_continuously': False,
             'rtol': 1e-06,
             'sensmethod': 'STAGGERED',
             'store_event_points': True,
             'suppress_alg': False,
             'suppress_sens': False,
             'time_limit': 0,
             'tout1': 0.0001,
             'usejac': False,
             'usesens': False,
             'verbosity': 30}
        
        return d

class _CVode:
    @staticmethod
    def default_options():
        d =  {'atol': np.array([]),
              'backward': False,
              'clock_step': False,
              'discr': 'BDF',
              'display_progress': True,
              'dqrhomax': 0.0,
              'dqtype': 'CENTERED',
              'external_event_detection': False,
              'inith': 0.0,
              'iter': 'Newton',
              'linear_solver': 'DENSE',
              'maxcor': 3,
              'maxcorS': 3,
              'maxh': 0.0,
              'maxkrylov': 5,
              'maxncf': 10,
              'maxnef': 20,
              'maxord': 5,
              'maxsteps': 500,
              'minh': 0.0,
              'nnz': -1,
              'norm': 'WRMS',
              'num_threads': 1,
              'pbar': [],
              'precond': "Banded",
              'report_continuously': False,
              'rtol': 1e-06,
              'sensmethod': 'STAGGERED',
              'stablimit': False,
              'store_event_points': True,
              'suppress_sens': False,
              'time_limit': 0,
              'usejac': False,
              'usesens': False,
              'verbosity': 30}
        d.pop("atol")
        return d
        
class _LSODAR(LSODAR):
    @staticmethod
    def default_options():
        d = {'atol': np.array([]),
             'backward': False,
             'clock_step': False,
             'display_progress': True,
             'maxh': 0.0,
             'maxordn': 12,
             'maxords': 5,
             'maxsteps': 100000,
             'num_threads': 1,
             'report_continuously': False,
             'rkstarter': 1,
             'rtol': 1e-06,
             'store_event_points': True,
             'time_limit': 0,
             'usejac': False,
             'verbosity': 30}
        d.pop("atol")
        return d
        
