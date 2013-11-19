"""test for odecomponent module"""

import unittest

import gotran
from modelparameters.logger import suppress_logging
from modelparameters.codegeneration import sympycode
from modelparameters.sympytools import sp_namespace

globals().update(sp_namespace)

from gotran.common import GotranException

from gotran.model.odecomponents2 import *
from sympy import Symbol, Derivative

suppress_logging()

class TestODEComponent(unittest.TestCase):

    def test_creation(self):

        # Adding a phoney ODE
        ode = ODE("test")

        # Add states and parameters
        j=ode.add_state("j", [1.0])
        i=ode.add_state("i", 2.0)
        k=ode.add_state("k", 3.0)

        ii=ode.add_parameter("ii", 0.0)
        jj=ode.add_parameter("jj", 0.0)
        kk=ode.add_parameter("kk", 0.0)

        self.assertEqual(ode.num_states, 3)
        self.assertEqual(ode.num_field_states, 1)
        self.assertEqual(ode.num_parameters, 3)

        # Add an Expression
        ode.alpha = i*j

        # Add derivatives for all states in the main component
        ode.add_comment("Some nice derivatives")
        ode.di_dt = ode.alpha + ii
        ode.dj_dt = -ode.alpha - jj
        ode.dk_dt = kk*k*ode.alpha

        self.assertEqual(ode.num_intermediates, 4)

        # Add a component with 2 states
        ode("jada").add_states(m=2.0, n=3.0, l=1.0, o=4.0)
        ode("jada").add_parameters(ll=1.0, mm=[2.0])

        # Define a state derivative
        ode("jada").dm_dt = ode("jada").ll - (ode("jada").m - ode.i)
        jada = ode("jada")

        # Test num_foo
        self.assertEqual(jada.num_states, 4)
        self.assertEqual(jada.num_parameters, 2)
        self.assertEqual(ode.num_states, 7)
        self.assertEqual(ode.num_parameters, 5)
        self.assertEqual(ode.num_components, 2)
        self.assertEqual(ode.num_field_states, 1)
        self.assertEqual(ode.num_field_parameters, 1)
        self.assertEqual(jada.num_components, 1)

        # Add expressions to the component
        jada.tmp = jada.ll*jada.m**2+3/i - ii*jj
        jada.tmp2 = ode.j*exp(jada.tmp)

        # Reduce state n
        jada.add_solve_state(jada.n, 1-jada.l-jada.m-jada.n)

        self.assertEqual(ode.num_intermediates, 8)

        # Create a derivative expression
        ode.add_comment("More funky objects")
        jada.tmp3 = jada.tmp2.diff(ode.t) + jada.n + jada.o
        jada.add_derivative(jada.l, ode.t, jada.tmp3)
        jada.add_algebraic(jada.o, jada.o**2-exp(jada.o)+2/jada.o)

        self.assertEqual(ode.num_intermediates, 15)
        self.assertEqual(ode.num_state_expressions, 7)
        self.assertTrue(ode.is_complete)
        self.assertEqual(ode.num_full_states, 6)

        # Add another component to test rates
        bada = ode("bada")
        bada.add_parameters(nn=5.0, oo=3.0, qq=1.0, pp=2.0)
        
        nada = bada.add_markov_model("nada")
        nada.add_states(("r", 3.0), ("s", 4.0), ("q", 1.0), ("p", 2.0))
        self.assertEqual(bada.num_parameters, 4)
        self.assertEqual(bada.num_states, 4)
        nada.p = 1 - nada.r - nada.s - nada.q
        
        self.assertEqual("".join(p.name for p in ode.parameters), "iijjkkllmmnnooppqq")
        self.assertEqual("".join(s.name for s in ode.states), "jiklmnorsqp")
        self.assertEqual("".join(s.name for s in ode.full_states), "jiklmorsq")
        self.assertFalse(ode.is_complete)

        nada.add_single_rate(nada.r, nada.s, 1.0)
        nada.add_single_rate(nada.s, nada.r, 2.0)

        nada.add_single_rate(nada.s, nada.q, 2.0)
        nada.add_single_rate(nada.q, nada.s, 1.0)

        nada.add_single_rate(nada.q, nada.p, 3.0)
        nada.add_single_rate(nada.p, nada.q, 4.0)
        
        nada.finalize()
        self.assertTrue(ode.is_complete)
        

if __name__ == "__main__":
    unittest.main()
