"""test for odecomponent module"""

import unittest

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
        i=ode.add_state("i", 2.0)
        j=ode.add_state("j", 0.0)
        k=ode.add_state("k", 0.0)

        ii=ode.add_parameter("ii", 0.0)
        jj=ode.add_parameter("jj", 0.0)
        kk=ode.add_parameter("kk", 0.0)

        # Add an Sxpression
        ode.alpha = i*j
        
        ode.di_dt = ode.alpha + ii
        ode.dj_dt = -ode.alpha - jj
        ode.dk_dt = kk*k*ode.alpha

        # Add a component
        jada = ode.add_component("jada")
        jada.add_states(l=1.0, m=2.0)
        ode["jada"].add_parameters(ll=1.0, mm=2.0)

        jada.dm_dt = jada.ll - (jada.m - ode.i)

        # Add expressions to the component
        jada.tmp = jada.ll*jada.m**2+3/i - ii*jj
        jada.tmp2 = ode.j*exp(jada.tmp)
        jada.tmp3 = jada.tmp2.diff(ode.t)
        jada.dl_dt = jada.tmp3
        

if __name__ == "__main__":
    unittest.main()
