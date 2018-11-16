"""
To use install assimulo which is a python wrapper of
the sundials solvers

conda install assimulo

"""
from .utils import suppress_stdout_stderr
# Assimulo imports
try:
    with suppress_stdout_stderr():
        from assimulo.solvers import CVode
        from assimulo.problem import Explicit_Problem
    has_sundials = True
except ImportError as ex:
    has_sundials = False
from .odesolver import Solver, ODESolverError

# Local imports
__all__ = ["SundialsSolver", "has_sundials", "SundialsNotInstalled"]


class SundialsNotInstalled(Exception):pass


class SundialsSolver(Solver):

    def __init__(self, ode, method="cvode", **options):

        # Check imports
        if not has_sundials:
            msg = ("Chosen backend is sundials, but sundials is "+
                   "not installed")
            raise SundialsNotInstalled(msg)

        # # Check method
        # msg = ("Method {} is not a sundials method. ".format(method)+\
        #        "Possible methods are {}".format(sundials_methods))
        # assert method in sundials_methods, msg
        # self._method = method

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
        # if self._method == "cvode":
        self._solver = CVode(self._problem)

        # Parse parameters to rhs
        self._solver.sw = self._model_params.tolist()
        self._solver.problem_info["switches"]=True

        # Set verbosity to 100 (i.e turn of printing) if not specified
        verbosity = self._options.pop("verbosity", 100)
        self._options["verbosity"] = verbosity

        # Set verbosity to 100 (i.e turn of printing) if not specified
        maxh = self._options.pop("maxh", 5e-4)
        self._options["maxh"] = maxh

        self._solver.options.update((k,v) for k,v in self._options.items() \
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
    def list_solver_options():

        # if method == "cvode":
        return _CVode.default_options()

    def _solve(self, tsteps, *args, **kwargs):
        """
        Solve the problem
        """

        self._create_solver()

        t_end = tsteps[-1]
        ncp = len(tsteps)
        ncp_list = tsteps
        # Construct problem
        t, y = self._solver.simulate(t_end, ncp, ncp_list)
        # t,y = self._solver.simulate(t_end)

        return t,y

if has_sundials:

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
