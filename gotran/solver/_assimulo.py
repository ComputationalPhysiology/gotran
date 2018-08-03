"""
To use install assimulo which is a python wrapper of
the sundials solvers

pip install assimulo

"""
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

        self._options = options
        

        Solver.__init__(self, ode, arguments="tsp",
                        additional_declarations=additional_declarations,
                        jacobian_declaration_template=jacobian_declaration_template,
                        **options)
        self._create_solver()
       

    def _create_solver(self):
        # Create problem
        self._problem = Explicit_Problem(self._rhs, self._y0)

        # Set Jacobian if used
        if self._jac is not None:
            self._problem.jac = self._jac
            self._options['usejac'] = True
        
        # Create the solver
        if self._method == "cvode":
            self._solver = CVode(self._problem)
        elif self._method == "ida":
            self._solver = IDA(self._problem)
        elif self._method == "radau5ode":
            self._solver = Radau5ODE(self._problem)
        else:
            self._solver =  LSODAR(self._problem)


        # Parse parameters to rhs
        self._solver.sw = self._model_params.tolist()
        self._solver.problem_info["switches"]=True

        # Set verbosity to 100 (i.e turn of printing) if not specified
        verbosity = self._options.pop("verbosity", 100)
        self._options["verbosity"] = verbosity

        # Set verbosity to 100 (i.e turn of printing) if not specified
        maxh = self._options.pop("maxh", 5e-4)
        self._options["maxh"] = maxh
            
        self._solver.options.update( (k,v) for k,v in self._options.items() \
                                     if k in list(self._solver.options.keys()))

    def _eval_monitored(self, time, res, params, values):
        self.module.monitor(time, res, params, values)

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

    def _solve(self, tsteps, *args, **kwargs):
        """
        Solve the problem
        """

        self._create_solver()
        t_end = tsteps[-1]
        ncp = len(tsteps)
        ncp_list = tsteps
        # Construct problem
        t,y = self._solver.simulate(t_end, ncp, ncp_list)
        # t,y = self._solver.simulate(t_end)

        return t,y
    
if has_assimulo:
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


additional_declarations = r"""
%init%{{
import_array();
%}}

%include <exception.i>
%feature("autodoc", "1");

// Typemaps
%typemap(in) (double* {rhs_name})
{{
  // Check type
  if (!PyArray_Check($input))
    SWIG_exception(SWIG_TypeError, "Numpy array expected");

  // Get PyArrayObject
  PyArrayObject *xa = (PyArrayObject *)$input;

  // Check data type
  if (!(PyArray_ISCONTIGUOUS(xa) && PyArray_TYPE(xa) == NPY_DOUBLE))
    SWIG_exception(SWIG_TypeError, "Contigous numpy array of doubles "
                  "expected. Make sure the numpy array is contiguous, "
                  "and uses dtype=float_.");

  // Check size of passed array
  if ( PyArray_SIZE(xa) != {num_states} )
    SWIG_exception(SWIG_ValueError, "Expected a numpy array of size: "
                                  "{num_states}, for the {rhs_name} argument.");
  
  $1 = (double *)PyArray_DATA(xa);
}}

%typemap(in) (const double* {states_name})
{{
  // Check type
  if (!PyArray_Check($input))
    SWIG_exception(SWIG_TypeError, "Numpy array expected");

  // Get PyArrayObject
  PyArrayObject *xa = (PyArrayObject *)$input;

  // Check data type
  if (!(PyArray_ISCONTIGUOUS(xa) && PyArray_TYPE(xa) == NPY_DOUBLE))
    SWIG_exception(SWIG_TypeError, "Contigous numpy array of doubles expected."
           " Make sure the numpy array is contiguous, and uses dtype=float_.");

  // Check size of passed array
  if ( PyArray_SIZE(xa) != {num_states} )
    SWIG_exception(SWIG_ValueError, "Expected a numpy array of size: "
                              "{num_states}, for the {states_name} argument.");
  
  $1 = (double *)PyArray_DATA(xa);
}}

%typemap(in) (double* {states_name})
{{
  // Check type
  if (!PyArray_Check($input))
    SWIG_exception(SWIG_TypeError, "Numpy array expected");

  // Get PyArrayObject
  PyArrayObject *xa = (PyArrayObject *)$input;

  // Check data type
  if (!(PyArray_ISCONTIGUOUS(xa) && PyArray_TYPE(xa) == NPY_DOUBLE))
    SWIG_exception(SWIG_TypeError, "Contigous numpy array of doubles expected."
           " Make sure the numpy array is contiguous, and uses dtype=float_.");

  // Check size of passed array
  if ( PyArray_SIZE(xa) != {num_states} )
    SWIG_exception(SWIG_ValueError, "Expected a numpy array of size: "
                              "{num_states}, for the {states_name} argument.");
  
  $1 = (double *)PyArray_DATA(xa);
}}

%typemap(in) (double* {parameters_name})
{{
  // Check type
  if (!PyArray_Check($input))
    SWIG_exception(SWIG_TypeError, "Numpy array expected");

  // Get PyArrayObject
  PyArrayObject *xa = (PyArrayObject *)$input;

  // Check data type
  if (!(PyArray_ISCONTIGUOUS(xa) && PyArray_TYPE(xa) == NPY_DOUBLE))
    SWIG_exception(SWIG_TypeError, "Contigous numpy array of doubles expected."
           " Make sure the numpy array is contiguous, and uses dtype=float_.");

  // Check size of passed array
  if ( PyArray_SIZE(xa) != {num_parameters} )
    SWIG_exception(SWIG_ValueError, "Expected a numpy array of size: "
             "{num_parameters}, for the {parameters_name} argument.");
  
  $1 = (double *)PyArray_DATA(xa);
  
}}

// The typecheck
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) double *
{{
    $1 = PyArray_Check($input) ? 1 : 0;
}}

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) const double *
{{
    $1 = PyArray_Check($input) ? 1 : 0;
}}

%pythoncode%{{
def {rhs_function_name}({args}):
    '''
    Evaluates the right hand side of the model

    Arguments:
    ----------
{args_doc}    
    '''
    import numpy as np

    {rhs_name} = np.zeros_like({states_name})

    if not isinstance(parameters, np.ndarray):
        parameters = np.array(parameters, dtype=np.float_)
    _{rhs_function_name}({args}, {rhs_name})
    return {rhs_name}

{python_code}
%}}

%rename (_{rhs_function_name}) {rhs_function_name};

{jacobian_declaration}

{monitor_declaration}
"""


jacobian_declaration_template = """
// Typemaps
%typemap(in) (double* {jac_name})
{{
  // Check type
  if (!PyArray_Check($input))
    SWIG_exception(SWIG_TypeError, "Numpy array expected");

  // Get PyArrayObject
  PyArrayObject *xa = (PyArrayObject *)$input;

  // Check data type
  if (!(PyArray_ISCONTIGUOUS(xa) && PyArray_TYPE(xa) == NPY_DOUBLE))
    SWIG_exception(SWIG_TypeError, "Contigous numpy array of doubles "
                  "expected. Make sure the numpy array is contiguous, "
                  "and uses dtype=float_.");

  // Check size of passed array
  if ( PyArray_SIZE(xa) != {num_states}*{num_states} )
    SWIG_exception(SWIG_ValueError, "Expected a numpy array of size: "
                                  "{num_states}*{num_states}, for the {jac_name} argument.");
  
  $1 = (double *)PyArray_DATA(xa);
}}

%pythoncode%{{
def {jacobian_function_name}({args}, {jac_name}=None):
    '''
    Evaluates the jacobian of the model

    Arguments:
    ----------
{args_doc}    
    {jac_name} : np.ndarray (optional)
        The computed result 
    '''
    import numpy as np

    {jac_name} = np.zeros({num_states}*{num_states}, dtype=np.float_)
       
    _{jacobian_function_name}({args}, {jac_name})
    {jac_name}.shape = ({num_states},{num_states})
    return {jac_name}
%}}

%rename (_{jacobian_function_name}) {jacobian_function_name};
"""
