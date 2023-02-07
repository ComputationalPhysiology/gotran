import os
from collections import OrderedDict

import numpy as np
from goss import jit
from goss import ODESystemSolver
from goss import Progress
from goss import RL1
from goss import set_log_level
from goss import Timer as GossTimer
from goss import timings
from goss import TRACE
from goss.cuda import CUDAODESystemSolver

from gotran import load_ode

set_log_level(TRACE)

global check_nans
check_nans = False


def get_field_values(num_nodes, field_parameters={}, field_states=[], dtype=np.float_):
    if len(field_parameters) == 0:
        field_params = None
    else:
        field_params = (
            np.concatenate(
                tuple(
                    np.reshape(
                        np.linspace(value[0], value[1], num_nodes),
                        (num_nodes, 1),
                    )
                    for value in list(field_parameters.values())
                ),
                axis=1,
            )
            .astype(dtype)
            .copy()
        )

    field_states = np.zeros(num_nodes * len(field_states), dtype=dtype)

    return field_params, field_states


def step_solver(solver, tstop, dt, field_states, what):
    solver.get_field_states(field_states)

    t = 0.0
    p = Progress(what, int(tstop / dt))
    num_nans = 0
    while t < tstop + 1e-6:
        solver.set_field_states(field_states)
        solver.forward(t, dt)
        solver.get_field_states(field_states)
        if check_nans:
            n = np.isnan(field_states).sum()
            if n > num_nans:
                print(t, n)
            num_nans = n
        t += dt
        p += 1

    return field_states


def run_goss(
    ode,
    num_nodes,
    field_parameters={},
    field_states=["V"],
    tstop=300,
    dt=0.1,
):
    # Create GOSS solver
    solver = ODESystemSolver(
        num_nodes,
        RL1(),
        jit(
            ode,
            field_parameters=list(field_parameters.keys()),
            field_states=field_states,
        ),
    )

    solver.set_num_threads(8)

    # Reset default
    solver.reset_default()

    field_parameters, field_states = get_field_values(
        num_nodes,
        field_parameters,
        field_states,
    )
    if field_parameters is not None:
        solver.set_field_parameters(field_parameters)

    what = "GOSS"
    timer = GossTimer(what)  # noqa: F841
    return step_solver(solver, tstop, dt, field_states, what)


def run_gpu(
    ode,
    num_nodes,
    field_parameters={},
    field_states=["V"],
    tstop=300,
    dt=0.1,
    double=True,
):
    params = CUDAODESystemSolver.default_parameters()
    params.solver = "rush_larsen"
    params.code.states.field_states = field_states
    params.code.parameters.field_parameters = list(field_parameters.keys())
    params.code.float_precision = "double" if double else "single"

    solver = CUDAODESystemSolver(num_nodes, ode, params=params)

    dtype = np.float64 if double else np.float32
    field_parameters, field_states = get_field_values(
        num_nodes,
        field_parameters,
        field_states,
        dtype=dtype,
    )
    if field_parameters is not None:
        solver.set_field_parameters(field_parameters)

    what = "GPU " + ("DOUBLE" if double else "SINGLE")
    timer = GossTimer(what)  # noqa: F841
    return step_solver(solver, tstop, dt, field_states, what)


if __name__ == "__main__":
    os.environ["GOMP_CPU_AFFINITY"] = "0-7"
    filename = "tentusscher_panfilov_2006_M_cell.ode"
    ode = load_ode(filename)
    field_parameters = OrderedDict()
    for param in ode.parameters:
        if param.name in ["g_to", "g_CaL"]:
            field_parameters[param.name] = [param.init, param.init * 0.01]

    num_nodes = 1024 * 8 * 8
    gpu_result_double = run_gpu(
        ode,
        num_nodes,
        field_parameters=field_parameters,
        double=True,
    )
    gpu_result_single = run_gpu(
        ode,
        num_nodes,
        field_parameters=field_parameters,
        double=False,
    )
    print("NUM NAN SINGLE: ", np.isnan(gpu_result_single).sum())
    goss_result = run_goss(ode, num_nodes, field_parameters=field_parameters)

    print(
        "NUM DIFF DOUBLE:",
        np.absolute((goss_result - gpu_result_double) > 1e-8).sum(),
    )
    print(
        "NUM DIFF SINGLE:",
        np.absolute((goss_result - gpu_result_single) > 1e-8).sum(),
    )
    print(
        "PERCENT REL DIFF SINGLE > 0.1%",
        (np.absolute((goss_result - gpu_result_single) / goss_result) > 1.0e-3).sum()
        * 1.0
        / len(goss_result)
        * 100,
        "%",
    )
    print(
        "PERCENT REL DIFF SINGLE > 1%",
        (np.absolute((goss_result - gpu_result_single) / goss_result) > 1.0e-2).sum()
        * 1.0
        / len(goss_result)
        * 100,
        "%",
    )
    print(
        "PERCENT REL DIFF SINGLE > 2%",
        (np.absolute((goss_result - gpu_result_single) / goss_result) > 2.0e-2).sum()
        * 1.0
        / len(goss_result)
        * 100,
        "%",
    )
    print(
        "PERCENT REL DIFF SINGLE > 3%",
        (np.absolute((goss_result - gpu_result_single) / goss_result) > 3.0e-2).sum()
        * 1.0
        / len(goss_result)
        * 100,
        "%",
    )
    print()
    t = timings(True)
    print(t.str(True))
    goss_timing = t.get("GOSS", "Total time")
    gpu_double_timing = t.get("GPU DOUBLE", "Total time")
    gpu_single_timing = t.get("GPU SINGLE", "Total time")

    print("SPEEDUPS DOUBLE PREC:", eval("{}/{}".format(goss_timing, gpu_double_timing)))
    print("SPEEDUPS SINGLE PREC:", eval("{}/{}".format(goss_timing, gpu_single_timing)))
