#!/usr/bin/env python3
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from modelparameters.codegeneration import latex
from modelparameters.logger import error
from modelparameters.logger import warning
from modelparameters.parameterdict import ParameterDict
from modelparameters.parameters import OptionParam
from modelparameters.parameters import Param
from modelparameters.parameters import ScalarParam

from gotran.codegeneration.compilemodule import compile_module
from gotran.common.options import parameters
from gotran.model.loadmodel import load_ode
from gotran.model.utils import DERIVATIVE_EXPRESSION
from gotran.model.utils import special_expression

__author__ = "Johan Hake (hake.dev@gmail.com)"
__date__ = "2013-03-13 -- 2015-06-24"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU LGPL Version 3.0 or later"

try:
    from scipy.optimize import root
except ImportError:
    root = None


def gotranrun(filename, params):

    # Copy of default parameters
    generation = parameters.generation.copy()

    # Set body parameters
    generation.code.body.representation = params.code.body_repr
    generation.code.body.use_cse = params.code.use_cse
    generation.code.body.optimize_exprs = params.code.optimize_exprs

    # Set what code we are going to generate and not
    for what_not in [
        "componentwise_rhs_evaluation",
        "forward_backward_subst",
        "linearized_rhs_evaluation",
        "lu_factorization",
        "jacobian",
    ]:
        generation.functions[what_not].generate = False

    # Always generate code for monitored expressions
    generation.functions.monitored.generate = True

    # If scipy is used to solve we generate RHS and potentially a jacobian
    if params.solver == "scipy":
        generation.functions.rhs.generate = True
        generation.functions.jacobian.generate = params.code.generate_jacobian
    else:
        generation.solvers[params.solver].generate = True

    # Compile executeable code from gotran ode
    model_arguments = params.model_arguments
    if len(model_arguments) == 1 and model_arguments[0] == "":
        model_arguments = []

    if len(model_arguments) % 2 != 0:
        error("Expected an even number of values for 'model_arguments'")

    arguments = dict()
    for arg_name, arg_value in [
        (model_arguments[i * 2], model_arguments[i * 2 + 1])
        for i in range(int(len(model_arguments) / 2))
    ]:

        arguments[arg_name] = arg_value

    ode = load_ode(filename, **arguments)

    # Check for DAE
    if ode.is_dae:
        error(
            "Can only integrate pure ODEs. {0} includes algebraic states "
            "and is hence a DAE.".format(ode.name),
        )

    # Get monitored and plot states
    plot_states = params.plot_y

    # Get x_values
    x_name = params.plot_x

    state_names = [state.name for state in ode.full_states]
    monitored_plot = [
        plot_states.pop(plot_states.index(name))
        for name in plot_states[:]
        if name not in state_names
    ]

    monitored = []
    all_monitored_names = []
    for expr in sorted(ode.intermediates + ode.state_expressions):
        if expr.name not in monitored:
            monitored.append(expr.name)
        all_monitored_names.append(expr.name)

    # Check valid monitored plot
    for mp in monitored_plot:
        if mp not in monitored:
            error(f"{mp} is not a state or intermediate in this ODE")

    # Check x_name
    if x_name not in ["time"] + monitored + state_names:
        error(
            "Expected plot_x to be either 'time' or one of the plotable "
            "variables, got {}".format(x_name),
        )

    # Logic if x_name is not 'time' as we then need to add the name to
    # either plot_states or monitored_plot
    if x_name != "time":
        if x_name in state_names:
            plot_states.append(x_name)
        else:
            monitored_plot.append(x_name)

    module = compile_module(ode, params.code.language, monitored, generation)

    parameter_values = params.parameters
    init_conditions = params.init_conditions

    if len(parameter_values) == 1 and parameter_values[0] == "":
        parameter_values = []

    if len(init_conditions) == 1 and init_conditions[0] == "":
        init_conditions = []

    if len(parameter_values) % 2 != 0:
        error("Expected an even number of values for 'parameters'")

    if len(init_conditions) % 2 != 0:
        error("Expected an even number of values for 'initial_conditions'")

    user_params = dict()
    for param_name, param_value in [
        (parameter_values[i * 2], parameter_values[i * 2 + 1])
        for i in range(int(len(parameter_values) / 2))
    ]:

        user_params[param_name] = float(param_value)

    user_ic = dict()
    for state_name, state_value in [
        (init_conditions[i * 2], init_conditions[i * 2 + 1])
        for i in range(int(len(init_conditions) / 2))
    ]:

        user_ic[state_name] = float(state_value)

    # Use scipy to integrate model
    t0 = 0.0
    t1 = params.tstop
    dt = params.dt

    rhs = module.rhs
    jac = module.compute_jacobian if params.code.generate_jacobian else None
    y0 = module.init_state_values(**user_ic)
    model_params = module.init_parameter_values(**user_params)

    # Check for steady state solve
    if params.steady_state.solve and root:
        result = root(
            rhs,
            y0,
            args=(0.0, model_params),
            jac=jac,
            method=params.steady_state.method,
            tol=params.steady_state.tol,
        )

        if result.success:
            y0 = result.x
            print(
                "Found stead state:",
                ", ".join(
                    f"{state.name}: {value:e}"
                    for value, state in zip(y0, ode.full_states)
                ),
            )
        else:
            warning(result.message)

    tsteps = np.linspace(t0, t1, int(t1 / dt) + 1)

    # What solver should we use
    if params.solver == "scipy":
        try:
            from scipy.integrate import odeint
        except Exception as e:
            error(f"Problem importing scipy.integrate.odeint. {e}")
        results = odeint(rhs, y0, tsteps, Dfun=jac, args=(model_params,))

    else:

        # Get generated forward method
        forward = getattr(module, "forward_" + params.solver)

        results = [y0]
        states = y0.copy()

        # Integrate solution using generated forward method
        for ind, t in enumerate(tsteps[:-1]):

            # Step solver
            forward(states, t, dt, model_params)
            results.append(states.copy())

    # Plot results
    if not (plot_states or monitored or params.save_results):
        return

    # allocate memory for saving results
    if params.save_results:
        save_results = np.zeros(
            (len(results), 1 + len(state_names) + len(all_monitored_names)),
        )
        all_monitor_inds = np.array(
            [monitored.index(monitor) for monitor in all_monitored_names],
            dtype=int,
        )
        all_results_header = ", ".join(["time"] + state_names + all_monitored_names)

    plot_inds = [module.state_indices(state) for state in plot_states]

    monitor_inds = np.array(
        [monitored.index(monitor) for monitor in monitored_plot],
        dtype=int,
    )
    monitored_get_values = np.zeros(len(monitored), dtype=np.float_)

    # Allocate memory
    plot_values = np.zeros((len(plot_states) + len(monitored_plot), len(results)))

    for ind, (time, res) in enumerate(zip(tsteps, results)):

        if plot_states:
            plot_values[: len(plot_states), ind] = res[plot_inds]
        if monitored_plot or params.save_results:
            module.monitor(res, time, model_params, monitored_get_values)
        if monitored_plot:
            plot_values[len(plot_states) :, ind] = monitored_get_values[monitor_inds]
        if params.save_results:
            save_results[ind, 0] = time
            save_results[ind, 1 : len(state_names) + 1] = res
            save_results[ind, len(state_names) + 1 :] = monitored_get_values[
                all_monitor_inds
            ]

    # Save data
    if params.save_results:
        np.savetxt(
            f"{params.basename}.csv",
            save_results,
            header=all_results_header,
            delimiter=", ",
        )

    # Plot data

    if not (plot_states + monitored_plot):
        return

    # Fixes for derivatives
    monitored_plot_updated = []
    for monitor in monitored_plot:
        expr, what = special_expression(monitor, ode)
        if what == DERIVATIVE_EXPRESSION:
            var, dep = expr.groups()
            if var in ode.present_ode_objects and dep in ode.present_ode_objects:
                monitored_plot_updated.append(f"d{var}/d{dep}")
            else:
                monitored_plot_updated.append(monitor)
        else:
            monitored_plot_updated.append(monitor)

    plot_items = plot_states + monitored_plot
    if x_name != "time":
        x_values = plot_values[plot_items.index(x_name)]
    else:
        x_values = tsteps

    plt.rcParams["lines.linewidth"] = 2
    # line_styles = cycle([c+s for s in ["-", "--", "-.", ":"]
    # for c in plt.rcParams["axes.color_cycle"]])
    line_styles = cycle(
        [
            c + s
            for s in ["-", "--", "-.", ":"]
            for c in ["b", "g", "r", "c", "m", "y", "k"]
        ],
    )

    plotted_items = 0
    for what, values in zip(plot_items, plot_values):
        if what == x_name:
            continue
        plotted_items += 1

        plt.plot(x_values, values, next(line_styles))

    if plotted_items > 1:
        plt.legend([f"$\\mathrm{{{latex(value)}}}$" for value in plot_items])
    elif plot_items:
        plt.ylabel(f"$\\mathrm{{{latex(plot_items[0])}}}$")

    plt.xlabel(f"$\\mathrm{{{latex(x_name)}}}$")
    plt.title(ode.name.replace("_", "\\_"))
    plt.show()


def main():
    import os
    import sys

    body_params = parameters.generation.code.body.copy()

    code_params = ParameterDict(
        language=OptionParam("C", ["Python", "C"]),
        body_repr=dict.__getitem__(body_params, "representation"),
        use_cse=dict.__getitem__(body_params, "use_cse"),
        optimize_exprs=dict.__getitem__(body_params, "optimize_exprs"),
        generate_jacobian=Param(
            False,
            description="Generate and use analytic " "jacobian when integrating.",
        ),
    )

    steady_state = ParameterDict(
        solve=Param(
            False,
            description="If true scipy.optimize.root is used "
            "to find a steady state for a given parameters.",
        ),
        method=OptionParam(
            "hybr",
            [
                "hybr",
                "lm",
                "broyden1",
                "broyden2",
                "anderson",
                "linearmixing",
                "diagbroyden",
                "excitingmixing",
                "krylov",
            ],
        ),
        tol=ScalarParam(1e-5, description="Tolerance for root finding algorithm."),
    )

    solver = OptionParam(
        "scipy",
        ["scipy"] + list(parameters.generation.solvers.keys()),
        description="The solver that will be used to " "integrate the ODE.",
    )

    params = ParameterDict(
        solver=solver,
        steady_state=steady_state,
        parameters=Param([""], description="Set parameter of model"),
        init_conditions=Param([""], description="Set initial condition of model"),
        tstop=ScalarParam(100.0, gt=0, description="Time for stopping simulation"),
        dt=ScalarParam(0.1, gt=0, description="Timestep for plotting."),
        plot_y=Param(["V"], description="States or monitored to plot on the y axis."),
        plot_x=Param(
            "time",
            description="Values used for the x axis. Can be time "
            "and any valid plot_y variable.",
        ),
        model_arguments=Param([""], description="Set model arguments of the model"),
        code=code_params,
        save_results=Param(
            False,
            description="If True the results will be " "saved to a 'results.csv' file.",
        ),
        basename=Param(
            "results",
            description="The basename of the results "
            "file if the 'save_results' options is True.",
        ),
    )

    params.parse_args(usage="usage: %prog FILE [options]")

    if len(sys.argv) < 2:
        error("Expected a single gotran file argument.")

    if not os.path.isfile(sys.argv[1]):
        error("Expected the argument to be a file.", exception=IOError)

    file_name = sys.argv[1]
    gotranrun(file_name, params)


if __name__ == "__main__":
    main()
