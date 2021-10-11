"""test for odecomponent module"""

# Imports for evaluation of generated code
from __future__ import division
import numpy as np
from pathlib import Path

import pytest
import itertools
import importlib
import gotran

from gotran.common import parameters

_here = Path(__file__).absolute().parent


default_params = parameters["generation"].copy()
default_params.functions.jacobian.generate = True
default_params.functions.monitored.generate = False

state_repr_opts = sorted(
    dict.__getitem__(default_params.code.states, "representation")._options
)
param_repr_opts = dict.__getitem__(
    default_params.code.parameters, "representation"
)._options
body_repr_opts = dict.__getitem__(default_params.code.body, "representation")._options
body_optimize_opts = dict.__getitem__(
    default_params.code.body, "optimize_exprs"
)._options


def get_indexed(comp, name):
    gotran.codecomponent.check_arg(comp, gotran.CodeComponent)
    return [
        expr
        for expr in comp.body_expressions
        if isinstance(expr, gotran.IndexedExpression) and expr.basename == name
    ]


@pytest.fixture(scope="session")
def module():
    # Load reference CodeComponents
    ode = gotran.load_ode(_here.joinpath("tentusscher_2004_mcell_updated.ode"))

    # Options for code generation

    # Python code generator with default code generation paramters
    codegen = gotran.PythonCodeGenerator(default_params)

    module_code = codegen.module_code(ode)
    spec = importlib.util.spec_from_loader("ode", loader=None)
    _module = importlib.util.module_from_spec(spec)
    exec(module_code, _module.__dict__)

    # debug = 0

    # if debug:
    #     with open("module_code.py", "w") as f:
    #         f.write(module_code)
    return _module


@pytest.mark.parametrize(
    "body_repr, body_optimize, param_repr, state_repr, use_cse, float_precision",
    itertools.product(
        body_repr_opts,
        body_optimize_opts,
        param_repr_opts,
        state_repr_opts,
        [False, True],
        ["double", "single"],
    ),
)
def test_codegeneration(
    body_repr, body_optimize, param_repr, state_repr, use_cse, float_precision, module
):
    parameters_name = "parameters"
    states_name = "states"
    body_array_name = "body"
    body_in_arg = False
    # The test that will be attached to the TestCase class below

    # test_name = """
    # body repr:       {0},
    # body opt:        {1},
    # state repr:      {2},
    # parameter repr:  {3},
    # use_cse:         {4},
    # float_precision: {5},
    # state_name:      {6},
    # param_name:      {7},
    # body_name:       {8},
    # body_in_arg:     {9}""".format(
    #     body_repr,
    #     body_optimize,
    #     state_repr,
    #     param_repr,
    #     use_cse,
    #     float_precision,
    #     states_name,
    #     parameters_name,
    #     body_array_name,
    #     body_in_arg,
    # )
    # print("\nTesting code generation with parameters: " + test_name)

    states_values = module.init_state_values()
    parameter_values = module.init_parameter_values()
    rhs_ref_values = module.rhs(states_values, 0.0, parameter_values)
    jac_ref_values = module.compute_jacobian(states_values, 0.0, parameter_values)

    gen_params = parameters["generation"].copy()

    # Update code_params
    code_params = gen_params["code"]
    code_params["body"]["optimize_exprs"] = body_optimize
    code_params["body"]["representation"] = body_repr
    code_params["body"]["in_signature"] = body_in_arg
    code_params["body"]["array_name"] = body_array_name
    code_params["parameters"]["representation"] = param_repr
    code_params["parameters"]["array_name"] = parameters_name
    code_params["states"]["representation"] = state_repr
    code_params["states"]["array_name"] = states_name
    code_params["body"]["use_cse"] = use_cse
    code_params["float_precision"] = float_precision
    code_params["default_arguments"] = "stp"

    # Reload ODE for each test
    ode = gotran.load_ode(_here.joinpath("tentusscher_2004_mcell_updated.ode"))
    codegen = gotran.PythonCodeGenerator(gen_params)
    rhs_comp = gotran.rhs_expressions(ode, params=code_params)
    rhs_code = codegen.function_code(rhs_comp)

    rhs_namespace = {}
    exec(rhs_code, rhs_namespace)

    # DEBUG
    # if debug:
    #     test_name = "_".join(
    #         [
    #             body_repr,
    #             body_optimize,
    #             state_repr,
    #             param_repr,
    #             float_precision,
    #             states_name,
    #             parameters_name,
    #             body_array_name,
    #             str(body_in_arg),
    #         ]
    #     )
    #     test_name += "_use_cse_" + str(use_cse)

    #     with open("rhs_code_{0}.py".format(test_name), "w") as f:
    #         f.write(rhs_code)

    args = [states_values, 0.0]
    if param_repr != "numerals":
        args.append(parameter_values)

    if body_in_arg:
        body = np.zeros(rhs_comp.shapes[body_array_name])
        args.append(body)

    # Call the generated rhs function
    rhs_values = rhs_namespace["rhs"](*args)

    rhs_norm = np.sqrt(np.sum(rhs_ref_values - rhs_values) ** 2)

    # DEBUG
    # if debug:
    #     print("rhs norm:", rhs_norm)

    eps = 1e-8 if float_precision == "double" else 1e-6
    assert rhs_norm < eps

    # Only evaluate jacobian if using full body_optimization and body repr is reused_array
    if (
        body_optimize != "numerals_symbols"
        and body_repr != "reused_array"
        and param_repr == "named"
    ):
        return

    jac_comp = gotran.jacobian_expressions(ode, params=code_params)
    jac_code = codegen.function_code(jac_comp)

    # if debug:
    #     test_name = "_".join(
    #         [
    #             body_repr,
    #             body_optimize,
    #             state_repr,
    #             param_repr,
    #             float_precision,
    #             states_name,
    #             parameters_name,
    #             body_array_name,
    #             str(body_in_arg),
    #         ]
    #     )
    #     test_name += "_use_cse_" + str(use_cse)

    #     open("jac_code_{0}.py".format(test_name), "w").write(jac_code)

    jac_namespace = {}
    exec(jac_code, jac_namespace)

    args = [states_values, 0.0]
    if param_repr != "numerals":
        args.append(parameter_values)

    if body_in_arg:
        body = np.zeros(jac_comp.shapes[body_array_name])
        args.append(body)

    jac_values = jac_namespace["compute_jacobian"](*args)
    jac_norm = np.sqrt(np.sum(jac_ref_values - jac_values) ** 2)

    # if debug:
    #     print("jac norm:", jac_norm)
    #     # print "JAC", jac_values

    eps = 1e-8 if float_precision == "double" else 1e-3
    assert jac_norm < eps


def test_generate_same_code():
    "Test that different ways of setting parameters generates the same code"

    # Generate basic code
    def generate_code(gen_params=None):
        ode = gotran.load_ode(_here.joinpath("tentusscher_2004_mcell_updated.ode"))
        code_params = None if gen_params is None else gen_params.code
        codegen = gotran.PythonCodeGenerator(gen_params)
        jac = gotran.jacobian_expressions(ode, params=code_params)

        comps = [
            gotran.rhs_expressions(ode, params=code_params),
            gotran.monitored_expressions(
                ode, ["i_NaK", "i_NaCa", "i_CaL", "d_fCa"], params=code_params
            ),
            gotran.componentwise_derivative(ode, 15, params=code_params),
            gotran.linearized_derivatives(ode, params=code_params),
            jac,
            gotran.diagonal_jacobian_expressions(jac, params=code_params),
            gotran.jacobian_action_expressions(jac, params=code_params),
        ]

        return [codegen.function_code(comp) for comp in comps]

    # Reset main parameters
    parameters["generation"].update(default_params)

    # Generate code based on default parameters
    default_codes = generate_code()

    gen_params = default_params.copy()

    # Update code_params
    code_params = gen_params["code"]

    # Update code parameters
    code_params["body"]["optimize_exprs"] = "none"
    code_params["body"]["representation"] = "array"
    code_params["body"]["array_name"] = "JADA"
    code_params["parameters"]["representation"] = "named"
    code_params["states"]["representation"] = "array"
    code_params["states"]["array_name"] = "STATES"
    code_params["body"]["use_cse"] = False
    code_params["float_precision"] = "double"

    # Code from updated parameters
    updated_codes = generate_code(gen_params)
    for updated_code, default_code in zip(updated_codes, default_codes):
        assert updated_code != default_code

    default_codes2 = generate_code(default_params)
    for default_code2, default_code in zip(default_codes2, default_codes):
        assert default_code2 == default_code


# for param_name, state_name, body_name in [
#     ["PARAMETERS", "STATES", "ALGEBRAIC"],
#     ["params", "states", "algebraic"],
# ]:
#     # for use_cse in [False, True]:
#     for use_cse in [True]:
#         for body_repr in ["array"]:  # ["array", "reused_array"]:
#             for body_in_arg in [True]:

#                 test_name = "_".join(
#                     [param_name, body_name, body_repr, str(body_in_arg)]
#                 )
#                 test_name += "_use_cse_" + str(use_cse)
#                 setattr(
#                     TestCodeComponent,
#                     "test_" + test_name,
#                     function_closure(
#                         body_repr,
#                         "none",
#                         "array",
#                         "array",
#                         use_cse,
#                         "double",
#                         param_name,
#                         state_name,
#                         body_name,
#                         body_in_arg,
#                     ),
#                 )


# # Populate the test class with methods
# for test_name, test_function in list(test_map.items()):
#     print(test_name, test_function)
#     setattr(TestCodeComponent, test_name, test_function)

# if __name__ == "__main__":
#     unittest.main()
