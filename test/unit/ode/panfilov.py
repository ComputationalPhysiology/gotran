__author__ = "Johan Hake (hake.dev@gmail.com)"
__date__ = "2012-05-07 -- 2012-10-22"
__copyright__ = "Copyright (C) 2012 " + __author__
__license__  = "GNU LGPL Version 3.0 or later"

import unittest
from gotran2 import *

class Creation(unittest.TestCase):

    def setUp(self):
        # Store ode
        ode = ODE("panfilov")
        ode.clear()
        
        # States
        e = ode.add_state("e", 0.0)
        g = ode.add_state("g", 0.0)
        
        # parameters
        v_rest = ode.add_parameter("v_rest", -85.0)
        v_peak = ode.add_parameter("v_peak", 35.0)
        time_constant = ode.add_parameter("time_constant", 1.0)
        
        # Local Python variables
        ode.a = 0.1
        ode.gs = 8.0
        ode.ga = ode.gs
        ode.M1 = 0.07
        ode.M2 = 0.3
        ode.eps1 = 0.01
        
        # Local compuations
        ode.E = (ode.e-ode.v_rest)/(ode.v_peak-ode.v_rest)
        ode.eps = ode.eps1 + ode.M1*ode.g/(ode.e+ode.M2)
        
        # Time differentials
        ode.diff(e, -ode.time_constant*(ode.v_peak-ode.v_rest)*\
                 (ode.ga*ode.E*(ode.E-ode.a)*(ode.E-1) + ode.E*ode.g))
        ode.diff(g, 0.25*ode.time_constant*ode.eps*(-g - ode.gs*e*(ode.E-ode.a-1)))
        
        assert(ode.is_complete)
        self.ode = ode


    def test_load_and_equality(self):
        """
        Test ODE loading from file and its equality with python created ones
        """

        ode = load_ode("panfilov")
        self.assertTrue(ode == self.ode)
        self.assertNotEqual(id(ode), id(self.ode))
        
        ode = load_ode("panfilov", small_change=True)

        # FIXME: Comment in when comparison works
        #self.assertFalse(ode == self.ode)

    def test_attributes(self):
        """
        Test ODE definition using attributes
        """
        ode = ODE("panfilov2")
        ode.clear()
        
        # States
        ode.add_state("e", 0.0)
        ode.add_state("g", 0.0)
        
        # parameters
        ode.add_parameter("v_rest", -85.0)
        ode.add_parameter("v_peak", 35.0)
        ode.add_parameter("time_constant", 1.0)
        
        # Local Python variables
        a = 0.1
        gs = 8.0
        ga = gs
        M1 = 0.07
        M2 = 0.3
        eps1 = 0.01
        
        # Local compuations
        E = (ode.e-ode.v_rest)/(ode.v_peak-ode.v_rest)
        eps = eps1 + M1*ode.g/(ode.e+M2)
        
        ode.diff(ode.e, -ode.time_constant*(ode.v_peak-ode.v_rest)*\
                 (ga*E*(E-a)*(E-1) + E*ode.g))
        ode.diff(ode.g, 0.25*ode.time_constant*eps*(-ode.g - gs*ode.e*(E-a-1)))

        self.assertTrue(ode == self.ode)

    def test_completness(self):
        """
        Test copletness of an ODE
        """
        self.assertTrue(self.ode.is_complete)
        
        ode = ODE("panfilov")
        self.assertTrue(ode.is_empty)
        
    def test_members(self):
        """
        Test that ODE has the correct members
        """
        ode = self.ode
        self.assertTrue(all(ode.has_state(state) for state in ["e", "g"]))
        self.assertTrue(all(ode.has_parameter(param) for param in \
                            ["v_rest", "v_peak", "time_constant"]))
        
    def test_python_code_gen(self):
        """
        Test generation of code
        """

        import numpy as np
        from gotran2.codegeneration.codegenerator import \
             CodeGenerator, ODERepresentation
        from gotran2.codegeneration.compilemodule import jit
        
        keep, use_cse, numerals, use_names = (1,0,0,1)

        gen = CodeGenerator(ODERepresentation(self.ode,
                                              keep_intermediates=keep, \
                                              use_cse=use_cse,
                                              parameter_numerals=numerals,\
                                              use_names=use_names))

        exec(gen.init_states_code())
        exec(gen.init_param_code())
        exec(gen.dy_code())

        parameters = default_parameters()
        states = init_values()
        dy_jit = np.asarray(states).copy()
        dy_correct = rhs(0.0, states, parameters)

        for keep, use_cse, numerals, use_names in \
                [(1,0,0,1), (1,0,0,0), \
                 (1,0,1,1), (1,0,1,0), \
                 (0,0,0,1), (0,0,0,0), \
                 (0,0,1,1), (0,0,1,0), \
                 (0,1,0,1), (0,1,0,0), \
                 (0,1,1,1), (0,1,1,0)]:

            oderepr = ODERepresentation(self.ode,
                                        keep_intermediates=keep, \
                                        use_cse=use_cse,
                                        parameter_numerals=numerals,\
                                        use_names=use_names)

            gen = CodeGenerator(oderepr)
            jit_oderepr = jit(oderepr)

            # Execute code
            exec(gen.dy_code())
            if numerals:
                dy_eval = rhs(0.0, states)
                jit_oderepr.rhs(0.0, states, dy_jit)
            else:
                dy_eval = rhs(0.0, states, parameters)
                jit_oderepr.rhs(0.0, states, parameters, dy_jit)

            self.assertTrue(np.sum(np.abs((dy_eval-dy_correct))) < 1e-12)
            self.assertTrue(np.sum(np.abs((dy_jit-dy_correct))) < 1e-12)
            
            
    def test_matlab_python_code(self):
        from gotran2.codegeneration.codegenerator import \
             MatlabCodeGenerator, ODERepresentation
        
        keep, use_cse, numerals, use_names = (1,0,0,1)

        gen = MatlabCodeGenerator(ODERepresentation(self.ode,
                                                    keep_intermediates=keep, \
                                                    use_cse=use_cse,
                                                    parameter_numerals=numerals,\
                                                    use_names=use_names))

        print gen.default_value_code()
        print gen.dy_code()
        
if __name__ == "__main__":
    unittest.main()
