"""test for odeobjects module"""

import pytest
from modelparameters.sympy import Symbol
from modelparameters.sympytools import symbols_from_expr

import gotran
from gotran.model.utils import ode_primitives

# from modelparameters.logger import suppress_logging
# from modelparameters.codegeneration import sympycode
# from gotran.model.odeobjects import *
# from gotran.model.expressions import StateDerivative

# import unittest


# from gotran.common import GotranException


def test_odeobjects():
    with pytest.raises(TypeError) as cm:
        gotran.ODEObject(45)

    # breakpoint()
    assert cm.value.args == (
        "expected 'str' (got '45' which "
        "is 'int') as the first argument while instantiating"
        " 'ODEObject'",
    )

    with pytest.raises(gotran.GotranException) as cm:
        gotran.ODEObject("_jada")
    assert cm.value.args == (
        "No ODEObject names can start " "with an underscore: '_jada'",
    )

    obj0 = gotran.ODEObject("jada bada")
    assert str(obj0) == "jada bada"

    obj1 = gotran.ODEObject("jada bada")

    assert obj0 != obj1

    obj0.rename("bada jada")
    assert str(obj0) == "bada jada"


def test_odevalueobjects():
    with pytest.raises(TypeError) as cm:
        gotran.ODEValueObject("jada", "bada")

    with pytest.raises(gotran.GotranException) as cm:
        gotran.ODEValueObject("_jada", 45)
    assert cm.value.args == (
        "No ODEObject names can start " "with an underscore: '_jada'",
    )

    obj = gotran.ODEValueObject("bada", 45)

    assert (
        Symbol(obj.name, real=True, imaginary=False, commutative=True, hermitian=True)
        == obj.sym
    )
    assert 45 == obj.value


def test_state():
    t = gotran.Time("t")

    with pytest.raises(TypeError) as cm:
        gotran.State("jada", "bada", t)

    with pytest.raises(gotran.GotranException) as cm:
        gotran.State("_jada", 45, t)
    assert cm.value.args == (
        "No ODEObject names can start " "with an underscore: '_jada'",
    )

    s = gotran.State("s", 45.0, t)
    a = gotran.State("a", 56.0, t)
    b = gotran.State("b", 40.0, t)

    s_s = s.sym
    a_s = a.sym
    b_s = b.sym
    t_s = t.sym

    # Create expression from just states symbols
    assert ode_primitives(s_s**2 * a_s + t_s * b_s * a_s, t_s) == set(
        [s_s, a_s, b_s, t_s],
    )

    # Create composite symbol
    sa_s = Symbol("sa")(s_s, a_s)

    assert symbols_from_expr(sa_s * a_s + t_s * b_s * a_s) == set([sa_s, a_s, b_s, t_s])

    # Test derivative
    assert gotran.StateDerivative(s, 1.0).sym == Symbol("s")(t_s).diff(t_s)


def test_param():
    with pytest.raises(TypeError) as cm:
        gotran.Parameter("jada", "bada")

    with pytest.raises(gotran.GotranException) as cm:
        gotran.Parameter("_jada", 45)
    assert cm.value.args == (
        "No ODEObject names can start " "with an underscore: '_jada'",
    )

    s = gotran.Parameter("s", 45.0)

    from gotran.model.utils import ode_primitives

    t = gotran.Time("t")
    a = gotran.State("a", 56.0, t)
    b = gotran.State("b", 40.0, t)

    s_s = s.sym
    a_s = a.sym
    b_s = b.sym
    t_s = t.sym

    # Create expression from just states symbols
    assert ode_primitives(s_s**2 * a_s + t_s * b_s * a_s, t_s) == set(
        [s_s, a_s, b_s, t_s],
    )

    # Create composite symbol
    sa_s = Symbol(
        "sa",
        real=True,
        imaginary=False,
        commutative=True,
        hermitian=True,
        complex=True,
    )(s_s, a_s)
    sa_s._assumptions["real"] = True
    sa_s._assumptions["commutative"] = True
    sa_s._assumptions["imaginary"] = False
    sa_s._assumptions["hermitian"] = True

    assert symbols_from_expr(sa_s * a_s + t_s * b_s * a_s) == set([sa_s, a_s, b_s, t_s])
