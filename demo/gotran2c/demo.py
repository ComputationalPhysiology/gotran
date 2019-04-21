import math
import numpy as np
import ctypes
from ctypes import c_int, c_long, c_ulong, c_double
import time
import os
import matplotlib.pyplot as plt


libname = 'libdemo.so'
libdir = '.'
libpath = os.path.join(libdir, libname)

assert os.path.isfile(libpath)
libbase_model = np.ctypeslib.load_library(libname, libdir)

# Need to hardcode this
num_parameters = 70
num_states = 19


def init_lib():
    """
    Make sure that arrays passed to C is of the correct types.
    """

    float64_array = np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                           flags="contiguous")
    float64_array_2d = np.ctypeslib.ndpointer(dtype=c_double, ndim=2,
                                              flags="contiguous")

    libbase_model.init_state_values.restype = None # void
    libbase_model.init_state_values.argtypes = [
        float64_array
    ]

    libbase_model.init_parameters_values.restype = None # void
    libbase_model.init_parameters_values.argtypes = [
        float64_array
    ]

    solve_functions = [
        libbase_model.ode_solve_forward_euler,
        libbase_model.ode_solve_rush_larsen,
    ]

    for func in solve_functions:
        func.restype = None # void
        func.argtypes = [
            float64_array,     # u
            float64_array,     # parameters
            float64_array_2d,  # u_values
            float64_array,     # t_values
            c_int,             # num_timesteps
            c_double,          # dt
        ]


def init_parameters():
    parameters = np.zeros(num_parameters, dtype=np.float64)
    libbase_model.init_parameters_values(parameters)
    return parameters


def solve(t_start, t_end, dt, num_steps=None, method='fe'):
    parameters = init_parameters()

    if type(dt) is not float:
        dt = float(dt)
    if num_steps is not None:
        assert type(num_steps) is int
        t_end = dt*num_steps
    else:
        num_steps = round((t_end-t_start)/dt)

    t_values = np.linspace(t_start, t_end, num_steps + 1)

    u = np.zeros(num_states, dtype=np.float64)

    libbase_model.init_state_values(u)
    u_values = np.zeros((num_steps+1, u.shape[0]), dtype=np.float64)
    u_values[0, :] = u[:]

    if method == 'fe':
        libbase_model.ode_solve_forward_euler(u, parameters, u_values,
                                              t_values, num_steps, dt)
    elif method == 'rush_larsen':
        libbase_model.ode_solve_rush_larsen(u, parameters, u_values,
                                            t_values, num_steps, dt)
    else:
        raise ValueError('Invalid method %s' % method)

    return t_values, u_values


def main():
    
    t_start = 0.0
    t_end = 100.0
    dt = 0.001

    t_values, u_values = solve(
        t_start,
        t_end,
        dt,
        method='fe'
    )

    V_idx = libbase_model.state_index('V')

    fig, ax = plt.subplots()
    ax.plot(t_values, u_values[:, V_idx])
    ax.set_title('Membrane potential')
    ax.set_xlabel('Time (ms)')
    plt.show()


if __name__ == '__main__':
    init_lib()
    main()

    
