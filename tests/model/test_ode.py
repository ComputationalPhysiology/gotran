"""test for odecomponent module"""
# import unittest
import os
from pathlib import Path

import pytest
from modelparameters import ScalarParam
from modelparameters.codegeneration import sympycode
from modelparameters.sympytools import symbols_from_expr
from sympy import Matrix, exp, log

import gotran

_here = Path(__file__).absolute().parent


def test_creation():

    # Adding a phoney ODE
    ode = gotran.ODE("test")

    # Add states and parameters
    j = ode.add_state("j", 1.0)
    i = ode.add_state("i", 2.0)
    k = ode.add_state("k", 3.0)

    ii = ode.add_parameter("ii", 0.0)
    jj = ode.add_parameter("jj", 0.0)
    kk = ode.add_parameter("kk", 0.0)

    # Try overwriting state
    with pytest.raises(gotran.GotranException) as cm:
        ode.add_parameter("j", 1.0)

    # Try overwriting parameter
    with pytest.raises(gotran.GotranException) as cm:
        ode.add_state("ii", 1.0)

    assert ode.num_states == 3
    assert ode.num_parameters == 3
    assert ode.present_component == ode

    # Add an Expression
    ode.alpha = i * j

    # Add derivatives for all states in the main component
    ode.add_comment("Some nice derivatives and an algebraic expression")
    ode.di_dt = ode.alpha + ii
    ode.dj_dt = -ode.alpha - jj
    ode.alg_k_0 = kk * k * ode.alpha

    assert ode.num_intermediates == 1

    # Add a component with 2 states
    ode("jada").add_states(m=2.0, n=3.0, l=1.0, o=4.0)
    ode("jada").add_parameters(ll=1.0, mm=2.0)

    # Define a state derivative
    ode("jada").dm_dt = ode("jada").ll - (ode("jada").m - ode.i)

    jada = ode("jada")
    assert ode.present_component == jada

    # Test num_foo
    assert jada.num_states == 4
    assert jada.num_parameters == 2
    assert ode.num_states == 7
    assert ode.num_parameters == 5
    assert ode.num_components == 2
    assert jada.num_components == 1

    # Add expressions to the component
    jada.tmp = jada.ll * jada.m ** 2 + 3 / i - ii * jj
    jada.tmp2 = ode.j * exp(jada.tmp)

    # Reduce state n
    jada.add_solve_state(jada.n, 1 - jada.l - jada.m - jada.n)

    assert ode.num_intermediates == 4

    # Try overwriting parameter with expression
    with pytest.raises(gotran.GotranException) as cm:
        jada.ll = jada.tmp * jada.tmp2

    # Create a derivative expression
    ode.add_comment("More funky objects")
    jada.tmp3 = jada.tmp2.diff(ode.t) + jada.n + jada.o
    jada.add_derivative(jada.l, ode.t, jada.tmp3)
    jada.add_algebraic(jada.o, jada.o ** 2 - exp(jada.o) + 2 / jada.o)

    assert ode.num_intermediates == 9
    assert ode.num_state_expressions == 6
    assert ode.is_complete
    assert ode.num_full_states == 6

    # Try adding expressions to ode component
    with pytest.raises(gotran.GotranException) as cm:
        ode.p = 1.0

    # Check used in and dependencies for one intermediate
    tmp3 = ode.present_ode_objects["tmp3"]
    assert ode.object_used_in[tmp3] == {ode.present_ode_objects["dl_dt"]}

    for sym in symbols_from_expr(tmp3.expr, include_derivatives=True):
        assert (
            ode.present_ode_objects[sympycode(sym)] in ode.expression_dependencies[tmp3]
        )

    # Add another component to test rates
    bada = ode("bada")
    bada.add_parameters(nn=5.0, oo=3.0, qq=1.0, pp=2.0)

    nada = bada.add_component("nada")
    nada.add_states(("r", 3.0), ("s", 4.0), ("q", 1.0), ("p", 2.0))
    assert bada.num_parameters == 4
    assert bada.num_states == 4
    nada.p = 1 - nada.r - nada.s - nada.q

    assert "".join(p.name for p in ode.parameters) == "iijjkkllmmnnooppqq"
    assert "".join(s.name for s in ode.states) == "jiklmnorsqp"
    assert not ode.is_complete

    # Add rates to component making it a Markov model component
    nada.rates[nada.r, nada.s] = 3 * exp(-i)

    # Try add a state derivative to Markov model
    with pytest.raises(gotran.GotranException):
        nada.ds_dt = 3.0

    nada.rates[nada.s, nada.r] = 2.0
    nada.rates[nada.s, nada.q] = 2.0
    nada.rates[nada.q, nada.s] = 2 * exp(-i)
    nada.rates[nada.q, nada.p] = 3.0
    nada.rates[nada.p, nada.q] = 4.0

    assert ode.present_component == nada

    markov = bada.add_component("markov_2")
    markov.add_states(("tt", 3.0), ("u", 4.0), ("v", 1.0))

    with pytest.raises(gotran.GotranException):
        markov.rates[nada.s, nada.r] = 2.0

    with pytest.raises(gotran.GotranException):
        markov.rates[markov.tt, markov.u] = 2 * exp(markov.u)

    with pytest.raises(gotran.GotranException):
        markov.rates[[markov.tt, markov.u, markov.v]] = 5.0

    with pytest.raises(gotran.GotranException):
        markov.rates[[markov.tt, markov.u, markov.v]] = Matrix(
            [[1, 2 * i, 0.0], [0.0, 2.0, 4.0]],
        )
    with pytest.raises(gotran.GotranException):
        markov.rates[markov.tt, markov.tt] = 5.0

    markov.rates[[markov.tt, markov.u, markov.v]] = Matrix(
        [[0.0, 2 * i, 2.0], [4.0, 0.0, 2.0], [5.0, 2.0, 0.0]],
    )

    ode.finalize()
    assert ode.is_complete
    assert ode.is_dae

    # Test Mass matrix
    vector = ode.mass_matrix * Matrix([1] * ode.num_full_states)
    assert (0, 0) == (vector[2], vector[5])
    assert sum(ode.mass_matrix) == ode.num_full_states - 2
    assert sum(vector) == ode.num_full_states - 2

    assert "".join(s.name for s in ode.full_states) == "ijkmlorsqttuv"
    assert ode.present_component == ode

    # Test saving
    ode.save("test_ode")

    # Test loading
    ode_loaded = gotran.load_ode("test_ode")

    # Clean
    os.unlink("test_ode.ode")

    # Test same signature
    # self.assertEqual(ode.signature(), ode_loaded.signature())

    # Check that all objects are the same and evaluates to same value
    for name, obj in list(ode.present_ode_objects.items()):
        loaded_obj = ode_loaded.present_ode_objects[name]
        assert type(obj) == type(loaded_obj)
        assert loaded_obj.param.value == pytest.approx(obj.param.value)


def test_extract_components():

    ode = gotran.load_ode(_here.joinpath("tentusscher_2004_mcell_updated.ode"))

    potassium = ode.extract_components(
        "Potassium",
        "Rapid time dependent potassium current",
        "Inward rectifier potassium current",
        "Slow time dependent potassium current",
        "Potassium pump current",
        "Potassium dynamics",
        "Transient outward current",
    )

    for name, obj in list(potassium.present_ode_objects.items()):
        orig_obj = ode.present_ode_objects[name]
        assert orig_obj.param.value == pytest.approx(obj.param.value)

    sodium = ode.extract_components(
        "Sodium",
        "Fast sodium current",
        "Sodium background current",
        "Sodium potassium pump current",
        "Sodium calcium exchanger current",
        "Sodium dynamics",
    )

    for name, obj in list(sodium.present_ode_objects.items()):
        orig_obj = ode.present_ode_objects[name]
        assert orig_obj.param.value == pytest.approx(obj.param.value)

    calcium = ode.extract_components(
        "Calcium",
        "Calcium dynamics",
        "Calcium background current",
        "Calcium pump current",
        "L type ca current",
    )

    for name, obj in list(calcium.present_ode_objects.items()):
        orig_obj = ode.present_ode_objects[name]
        assert orig_obj.param.value == pytest.approx(obj.param.value)


def test_subode():
    ode_from_file = gotran.load_ode(_here.joinpath("tentusscher_2004_mcell_updated"))

    ode = gotran.ODE("Tentusscher_2004_merged")

    # Add parameters and states
    ode.add_parameters(
        Na_i=ScalarParam(11.6),
        Na_o=ScalarParam(140),
        K_i=ScalarParam(138.3),
        K_o=ScalarParam(5.4),
        Ca_o=ScalarParam(2),
        Ca_i=ScalarParam(0.0002),
    )

    mem = ode("Membrane")
    mem.add_states(V=ScalarParam(-86.2))

    mem.add_parameters(
        Cm=ScalarParam(0.185),
        F=ScalarParam(96485.3415),
        R=ScalarParam(8314.472),
        T=ScalarParam(310),
        V_c=ScalarParam(0.016404),
        stim_amplitude=ScalarParam(0),
        stim_duration=ScalarParam(1),
        stim_period=ScalarParam(1000),
        stim_start=ScalarParam(1),
    )

    rev_pot = ode("Reversal potentials")
    rev_pot.add_parameter("P_kna", ScalarParam(0.03))

    # Add intermediates
    ode.i_Stim = (
        -mem.stim_amplitude
        * (1 - 1 / (1 + exp(5.0 * ode.t - 5.0 * mem.stim_start)))
        / (1 + exp(5.0 * ode.t - 5.0 * mem.stim_start - 5.0 * mem.stim_duration))
    )

    rev_pot.E_Na = mem.R * mem.T * log(ode.Na_o / ode.Na_i) / mem.F
    rev_pot.E_K = mem.R * mem.T * log(ode.K_o / ode.K_i) / mem.F
    rev_pot.E_Ks = (
        mem.R
        * mem.T
        * log(
            (ode.Na_o * rev_pot.P_kna + ode.K_o) / (ode.K_i + ode.Na_i * rev_pot.P_kna),
        )
        / mem.F
    )
    rev_pot.E_Ca = 0.5 * mem.R * mem.T * log(ode.Ca_o / ode.Ca_i) / mem.F

    # Get the E_Na expression and one dependency
    E_Na = ode.present_ode_objects["E_Na"]
    Na_i = ode.present_ode_objects["Na_i"]

    assert isinstance(Na_i, gotran.Parameter)

    # Check dependencies
    assert Na_i in ode.expression_dependencies[E_Na]
    assert E_Na in ode.object_used_in[Na_i]

    ode.import_ode(_here.joinpath("Sodium"))

    # Check dependencies after sub ode has been loaded
    E_Na = ode.present_ode_objects["E_Na"]
    Na_i_new = ode.present_ode_objects["Na_i"]
    assert isinstance(Na_i_new, gotran.State)

    assert Na_i not in ode.expression_dependencies[E_Na]
    assert Na_i_new in ode.expression_dependencies[E_Na]
    assert E_Na not in ode.object_used_in[Na_i]
    assert E_Na in ode.object_used_in[Na_i_new]

    pot = gotran.load_ode(_here.joinpath("Potassium"))
    ode.import_ode(pot, prefix="pot")

    pot_comps = [comp.name for comp in pot.components]

    # Add sub ode by extracting components
    ode.import_ode(
        ode_from_file,
        components=[
            "Calcium dynamics",
            "Calcium background current",
            "Calcium pump current",
            "L type ca current",
        ],
    )

    # Get all the Potassium currents
    i_K1 = ode("Inward rectifier potassium current").pot_i_K1
    i_Kr = ode("Rapid time dependent potassium current").pot_i_Kr
    i_Ks = ode("Slow time dependent potassium current").pot_i_Ks
    i_to = ode("Transient outward current").pot_i_to
    i_p_K = ode("Potassium pump current").pot_i_p_K

    # Get all the Sodium currents
    i_Na = ode("Fast sodium current").i_Na
    i_b_Na = ode("Sodium background current").i_b_Na
    i_NaCa = ode("Sodium calcium exchanger current").i_NaCa
    i_NaK = ode("Sodium potassium pump current").i_NaK

    # Get all the Calcium currents
    i_CaL = ode("L type ca current").i_CaL
    i_b_Ca = ode("Calcium background current").i_b_Ca
    i_p_Ca = ode("Calcium pump current").i_p_Ca

    # Membrane potential derivative
    mem.dV_dt = (
        -i_Ks
        - i_p_K
        - i_Na
        - i_K1
        - i_p_Ca
        - i_b_Ca
        - i_NaK
        - i_CaL
        - i_Kr
        - ode.i_Stim
        - i_NaCa
        - i_b_Na
        - i_to
    )

    # Finalize ODE
    ode.finalize()

    for name, obj in list(ode.present_ode_objects.items()):

        # If object in prefixed potassium components
        if ode.object_component[obj].name in pot_comps:
            loaded_obj = ode_from_file.present_ode_objects[name.replace("pot_", "")]
        else:
            loaded_obj = ode_from_file.present_ode_objects[name]

        assert type(obj) == type(loaded_obj)
        assert loaded_obj.param.value == pytest.approx(obj.param.value)
