__all__ = ["CUDAODESystemSolver", "ODECUDAHandler"]

from gotran import CUDACodeGenerator, get_solver_fn, parameters
from gotran.common import Timer

from modelparameters.parameterdict import *

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

import numpy as np

def get_float_type(code_params):
    return {'single': 'float32',
            'double': 'float64'}[code_params.float_precision]

def get_np_float_type(code_params):
    return {'single': np.float32,
            'double': np.float64}[code_params.float_precision]

class ODECUDAHandler(object):
    def __init__(self, num_nodes, ode):

        self._num_nodes = num_nodes
        self._ode = ode

        self._cuda_ready = False

    @staticmethod
    def default_parameters():
        # Start out with a modified subset of the global parameters
        return CUDAODESystemSolver.default_parameters().copy()

    @property
    def num_nodes(self):
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, value):
        if self.is_ready():
            raise Exception("Cannot change number of nodes while CUDA handler "
                            "is initialised.")
        else:
            self._num_nodes = value
            self.params.code.n_nodes = value

    def init_cuda(self, params=None):
        if self.is_ready():
            self.clean_up()

        params = params or {}
        self.params = self.default_parameters()
        self.params.update(params)

        # self.params.code.n_nodes = self._num_nodes

        ccg = CUDACodeGenerator(self.params)
        self._cuda_code = ccg.solver_code(self._ode, self.params.solver)

        dev = cuda.Device(0)
        comp_cap = dev.compute_capability()
        arch = 'sm_%d%d' % comp_cap
        optimisations = dict()

        self._mod = SourceModule(self._cuda_code, arch=arch)

        float_t = 'float64' if self.params.code.float_precision == 'double' \
                  else 'float32'
        float_sz = np.dtype(float_t).itemsize

        # Allocate and initialise states
        init_states_fn = self._mod.get_function('init_state_values')
        self._h_states = np.zeros(self._num_nodes*self._ode.num_states,
                                  dtype=float_t)
        self._d_states = \
            cuda.mem_alloc(float_sz*self._num_nodes*self._ode.num_states)
        field_states = self.params.code.states.field_states
        # FIXME: modelparameters needs a ListParam
        if len(field_states) == 1 and field_states[0] == "":
            field_states = list()
        self._d_field_states = None
        if len(field_states) > 0:
            self._d_field_states = \
                cuda.mem_alloc(float_sz*self._num_nodes*len(field_states))
        init_states_fn(self._d_states, block=self._get_block(),
                       grid=self._get_grid())
        cuda.memcpy_dtoh(self._h_states, self._d_states)

        # Allocate and initialise parameters
        _parameter_values = [parameter.init
                             for parameter in self._ode.parameters]
        self._h_parameters = np.array(_parameter_values, dtype=float_t)
        self._d_parameters = \
            cuda.mem_alloc(float_sz*len(self._h_parameters))
        field_parameters = self.params.code.parameters.field_parameters
        # FIXME: modelparameters needs a ListParam
        if len(field_parameters) == 1 and field_parameters[0] == "":
            field_parameters = list()
        self._d_field_parameters = None
        if len(field_parameters) > 0:
            init_fparams_fn = self._mod.get_function('init_field_parameters')
            self._d_field_parameters = \
                cuda.mem_alloc(float_sz*self._num_nodes*len(field_parameters))
            init_fparams_fn(self._d_field_parameters, block=self._get_block(),
                            grid=self._get_grid())
        cuda.memcpy_htod(self._d_parameters, self._h_parameters)

        # Set forward solver function
        solver_type = self.params.solver
        solver_function_name = self.params.solvers[solver_type].function_name
        self._forward_fn = \
            self._mod.get_function(solver_function_name)

        self._cuda_ready = True

    def clean_up(self):
        self._cuda_ready = False
        for _d_array in (self._d_states, self._d_parameters,
                         self._d_field_states, self._d_field_parameters):
            try:
                _d_array.free()
            except cuda.LogicError:
                continue

    def forward(self, t, dt, update_host_states=False, synchronize=True):
        if not self.is_ready():
            raise Exception('CUDA has not been initialised')
        else:
            timer = Timer("calculate CUDA forward")
            args = [self._d_states, t, dt, self._d_parameters]
            field_parameters = self.params.code.parameters.field_parameters
            # FIXME: modelparameters needs a ListParam
            if len(field_parameters) != 1 or field_parameters[0] != "":
                args.append(self._d_field_parameters)
            self._forward_fn(*args,
                             block=self._get_block(),
                             grid=self._get_grid())
            if synchronize:
                cuda.Context.synchronize()
            if update_host_states:
                timer = Timer("update host states")
                cuda.memcpy_dtoh(self._h_states, self._d_states)

    def is_ready(self):
        return self._cuda_ready

    def get_host_states(self):
        if not self.is_ready():
            raise Exception('CUDA has not been initialised')
        else:
            cuda.memcpy_dtoh(self._h_states, self._d_states)
            return self._h_states

    def get_host_parameters(self):
        if not self.is_ready():
            raise Exception('CUDA has not been initialised')
        else:
            cuda.memcpy_dtoh(self._h_parameters, self._d_parameters)

    def get_field_states(self, h_field_states):
        if not self.is_ready():
            raise Exception('CUDA has not been initialised')
        else:
            float_t = get_float_type(self.params.code)
            if str(h_field_states.dtype) != float_t:
                # TODO: ERROR!!
                pass
            get_field_states_fn = self._mod.get_function('get_field_states')
            timer = Timer("get_fs_fn")
            get_field_states_fn(self._d_states, self._d_field_states,
                                block=self._get_block(),
                                grid=self._get_grid())
            timer = Timer("get_fs_cpy")
            cuda.memcpy_dtoh(h_field_states, self._d_field_states)

    def set_field_states(self, h_field_states):
        if not self.is_ready():
            raise Exception('CUDA has not been initialised')
        else:
            float_t = get_float_type(self.params.code)
            if str(h_field_states.dtype) != float_t:
                # TODO: ERROR!!
                pass
            set_field_states_fn = self._mod.get_function('set_field_states')
            timer = Timer("set_fs_cpy")
            cuda.memcpy_htod(self._d_field_states, h_field_states)
            timer = Timer("set_fs_fn")
            set_field_states_fn(self._d_field_states, self._d_states,
                                block=self._get_block(),
                                grid=self._get_grid())

    def set_field_parameters(self, h_field_parameters):
        if not self.is_ready():
            raise Exception('CUDA has not been initialised')
        else:
            float_t = get_float_type(self.params.code)
            if str(h_field_parameters.dtype) != float_t:
                # TODO: ERROR!!
                pass
            cuda.memcpy_htod(self._d_field_parameters, h_field_parameters)

    def _get_block(self):
        return (min(self._num_nodes, self.params.block_size),
                1,
                1)

    def _get_grid(self):
        block_size = self.params.block_size
        return (self._num_nodes//block_size +
                    (0 if self._num_nodes % block_size == 0 else 1),
                1)

    def _get_code(self):
        return self._cuda_code if self.is_ready() else ''

class CUDAODESystemSolver(object):
    def __init__(self, num_nodes, ode, init_field_parameters=None,
                 params=None):
        # TODO: Check validity of arguments and params
        params = params or {}

        self.params = self.default_parameters()
        self.params.update(params)
        self._num_nodes = num_nodes
        self.params.code.n_nodes = num_nodes
        self._ode = ode
        self.runtimes = list()
        self.simulation_runtime = 0.0

        float_t = get_float_type(self.params.code)
        self.field_states = None
        p_field_states = params.code.states.field_states
        # FIXME: modelparameters needs a ListParam
        if len(p_field_states) > 0 and p_field_states[0] != "":
            self.field_states = np.zeros(
                self._num_nodes*len(p_field_states), dtype=float_t)
        p_field_parameters = params.code.parameters.field_parameters

        self._cudahandler = ODECUDAHandler(self._num_nodes,
                                           self._ode)
        self._cudahandler.init_cuda(params=params)

        if self.field_states is not None:
            self.get_field_states()
        # FIXME: modelparameters needs a ListParam
        if len(p_field_parameters) > 0 and p_field_states[0] != "":
            self.set_field_parameters(init_field_parameters)

    @staticmethod
    def default_parameters():
        # Start out with a modified subset of the global parameters
        default_params = CUDACodeGenerator.default_parameters().copy()
        return ParameterDict(
                code=default_params.code,
                solvers=default_params.solvers,
                solver=OptionParam(
                    "explicit_euler", default_params.solvers.keys(),
                    description="Default solver type"),
                block_size=ScalarParam(
                    1024, ge=1, description="Number of threads per CUDA block")
        )

    def _init_cuda(self, params=None):
        # TODO: Remove this and throw an error instead.
        # CUDA should be initialised in __init__ and never touched again.
        params = params or self.params
        self._cudahandler.init_cuda(params=params)

    def forward(self, t, dt, update_host_states=False,
                update_field_states=True,
                update_simulation_runtimes=True):
        if not self._cudahandler.is_ready():
            self._init_cuda() # TODO: Throw an error instead.

        float_t = get_np_float_type(self.params.code)
        t = float_t(t)
        dt = float_t(dt)

        if update_simulation_runtimes:
            now = time.time()

        if update_field_states and self.field_states is not None:
            self.set_field_states()
        self._cudahandler.forward(t, dt, update_host_states)
        if update_field_states and self.field_states is not None:
            self.get_field_states()

        if update_simulation_runtimes:
            self.runtimes.append((t, time.time() - now))
            self.simulation_runtime += self.runtimes[-1][1]

    def simulate(self, t0, dt, tstop, field_states_fn=None,
                 update_field_states=True, update_host_states=False):
        if field_states_fn is None:
            field_states_fn = lambda fstates: None

#        yield self.field_states if self.field_states is not None\
#              else None

        t = t0
        now1 = time.time()

        while t < tstop:
            now2 = time.time()

            self.forward(t, dt, update_host_states=update_host_states,
                         update_field_states=update_field_states,
                         update_simulation_runtimes=False)
            if update_field_states and self.field_states is not None:
                field_states_fn(self.field_states)

            self.runtimes.append((t, time.time() - now2))

            t += dt

#            yield self.field_states if self.field_states is not None\
#                  else None

        self.simulation_runtime = time.time() - now1

    def reset(self):
        self._cudahandler.clean_up()

    def get_field_states(self):
        timer = Timer("get field states")
        self._cudahandler.get_field_states(self.field_states)

    def set_field_states(self):
        timer = Timer("set field states")
        self._cudahandler.set_field_states(self.field_states)

    def set_field_parameters(self, field_parameters):
        self.field_parameters = field_parameters
        self._cudahandler.set_field_parameters(self.field_parameters)

    def get_cuda_states(self):
        return self._cudahandler.get_host_states()

    def get_cuda_parameters(self):
        return self._cudahandler.get_host_parameters()

    def get_cuda_code(self):
        return self._cudahandler._get_code()

    @property
    def num_nodes(self):
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, value):
        # TODO: Remove this. num_nodes should not be modifiable.
        if self._cudahandler.is_ready():
            raise Exception("Cannot change number of nodes while CUDA handler "
                            "is initialised.")
        else:
            self._num_nodes = value
            self.params.code.n_nodes = value