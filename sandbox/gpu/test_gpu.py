import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
from gotran import load_ode, ODERepresentation, CCodeGenerator

optimisations = dict()

filename = "tentusscher_panfilov_2006_M_cell_continuous"

ode = load_ode(filename)

# Get num states and parameters which sets the offset into the state
# and parameter array

num_states = ode.num_states
num_params = ode.num_parameters

oderepr = ODERepresentation(ode, **optimisations)
ccode = CCodeGenerator(oderepr)

init_state_code = ccode.init_states_code().replace(\
    "void", "__global__ void").replace(\
    "{","{\n  const int thread_ind = blockIdx.x*blockDim.x + threadIdx.x;"\
    "\n  const int offset = thread_ind*%d;" % num_states).replace("[", "[offset+")

init_param_code = ccode.init_param_code().replace(\
    "void", "__global__ void").replace(\
    "{","{\n  const int thread_ind = blockIdx.x*blockDim.x + threadIdx.x;"\
    "\n  const int offset = thread_ind*%d;" % num_params).replace("[", "[offset+")

rhs_code = ccode.dy_code(parameters_in_signature=True).replace(\
    "void", "__global__ void").replace(\
    "{","{\n  const int thread_ind = blockIdx.x*blockDim.x + threadIdx.x;"\
    "\n  const int state_offset = thread_ind*%d;"\
    "\n  const int param_offset = thread_ind*%d;" % (num_states, num_params)).replace(
    "states[", "states[state_offset+").replace(\
    "parameters[", "parameters[param_offset+")

gpu_code = "#include \"math.h\"\n\n" + init_state_code + "\n\n" + \
           init_param_code + "\n\n" + rhs_code

print gpu_code

# Can be compiled with nvcc -c gpu_code.cu -arch=sm_20
open("gpu_code.cu", "w").write(gpu_code)

# Fails Why?!?
mod = SourceModule(gpu_code, arch="sm_20")

