from pathlib import Path

import numpy as np
import pytest

from gotran.codegeneration import compile_module
from gotran.codegeneration import has_cppyy
from gotran.common.options import parameters
from gotran.input.cellml import cellml2ode

require_cppyy = pytest.mark.skipif(
    not has_cppyy(),
    reason="cppyy is required to run the test",
)


_here = Path(__file__).absolute().parent


@pytest.fixture(scope="session")
def ode():
    path = Path(_here.joinpath("tentusscher_noble_noble_panfilov_2004_a.cellml"))
    ode = cellml2ode(path)
    return ode


@pytest.fixture
def generation():
    return parameters.generation.copy()


@require_cppyy
def test_compile_rhs(ode, generation):
    # Compile ODE
    python_module = compile_module(ode, language="Python", generation_params=generation)
    c_module = compile_module(ode, language="C", generation_params=generation)

    assert np.isclose(
        python_module.init_state_values(),
        c_module.init_state_values(),
    ).all()
    assert np.isclose(
        python_module.init_parameter_values(),
        c_module.init_parameter_values(),
    ).all()
    states = c_module.init_state_values()
    parameters = c_module.init_parameter_values()

    rhs_python = python_module.rhs(states, 0, parameters)
    rhs_c = c_module.rhs(states, 0, parameters)
    assert np.isclose(rhs_python, rhs_c).all()

    assert python_module.parameter_indices("g_Kr") == c_module.parameter_indices("g_Kr")
    assert python_module.state_indices("V") == c_module.state_indices("V")


@require_cppyy
def test_compile_monitored(ode, generation):
    generation.functions.monitored.generate = True
    monitored = [i.name for i in ode.intermediates]
    python_module = compile_module(
        ode,
        monitored=monitored,
        language="Python",
        generation_params=generation,
    )
    c_module = compile_module(
        ode,
        monitored=monitored,
        language="C",
        generation_params=generation,
    )

    assert np.isclose(
        python_module.init_state_values(),
        c_module.init_state_values(),
    ).all()
    assert np.isclose(
        python_module.init_parameter_values(),
        c_module.init_parameter_values(),
    ).all()
    states = c_module.init_state_values()
    parameters = c_module.init_parameter_values()

    monitor_python = python_module.monitor(states, 0.0, parameters)
    monitor_c = c_module.monitor(states, 0, parameters)
    assert np.isclose(monitor_python, monitor_c).all()

    assert (
        python_module.monitor_indices("i_CaL")
        == c_module.monitor_indices("i_CaL")
        == monitored.index("i_CaL")
    )


@require_cppyy
def test_compile_jacobian(ode, generation):
    generation.functions.jacobian.generate = True

    python_module = compile_module(ode, language="Python", generation_params=generation)
    c_module = compile_module(ode, language="C", generation_params=generation)

    assert np.isclose(
        python_module.init_state_values(),
        c_module.init_state_values(),
    ).all()
    assert np.isclose(
        python_module.init_parameter_values(),
        c_module.init_parameter_values(),
    ).all()
    states = c_module.init_state_values()
    parameters = c_module.init_parameter_values()

    jac_c = c_module.compute_jacobian(states, 0, parameters)
    jac_python = python_module.compute_jacobian(states, 0.0, parameters).reshape(
        jac_c.shape,
    )
    assert np.isclose(jac_python, jac_c).all()
