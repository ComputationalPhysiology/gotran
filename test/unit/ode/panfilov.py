__author__ = "Johan Hake (hake.dev@gmail.com)"
__date__ = "2012-05-07 -- 2013-03-13"
__copyright__ = "Copyright (C) 2012 " + __author__
__license__  = "GNU LGPL Version 3.0 or later"

import unittest
from gotran import *

class Creation(unittest.TestCase):

    def setUp(self):
        # Store ode
        ode = ODE("panfilov")
        ode.clear()
        
        # States
        ode.add_state("e", 0.1)
        ode.add_state("g", 0.1)
        
        # parameters
        ode.add_parameter("v_rest", -85.0)
        ode.add_parameter("v_peak", 35.0)
        ode.add_parameter("time_constant", 1.0)
        
        # Local Python variables
        a = 0.1
        ode.gs = 8.0
        ode.ga = ode.gs
        ode.M1 = 0.07
        ode.M2 = 0.3
        ode.eps1 = 0.01
        
        # Local compuations
        ode.E = (ode.e-ode.v_rest)/(ode.v_peak-ode.v_rest)
        ode.eps = ode.eps1 + ode.M1*ode.g/(ode.e + ode.M2)
        
        # Time differentials
        ode.de_dt = -ode.time_constant*(ode.v_peak - ode.v_rest)*\
                 (ode.ga*ode.E*(ode.E - a)*(ode.E-1) + ode.E*ode.g)
        ode.dg_dt = 0.25*ode.time_constant*ode.eps*(-ode.g - ode.gs*ode.e*(ode.E-a-1))
        
        self.ode = ode
        assert(ode.is_complete)

    def test_load_and_equality(self):
        """
        Test ODE loading from file and its equality with python created ones
        """

        ode = load_ode("panfilov")
        
        self.assertTrue(ode == self.ode)
        self.assertNotEqual(id(ode), id(self.ode))
        
        ode = load_ode("panfilov", small_change=True)
        
        self.assertFalse(ode == self.ode)

    def test_completness(self):
        """
        Test copletness of an ODE
        """
        self.assertTrue(self.ode.is_complete)
        
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
        from gotran.codegeneration.codegenerator import \
             CodeGenerator, ODERepresentation
        from gotran.codegeneration.compilemodule import compile_module
        
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
        dy_correct = rhs(states, 0.0, parameters)

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
            jit_oderepr = compile_module(oderepr)

            # Execute code
            exec(gen.dy_code())
            if numerals:
                dy_eval = rhs(states, 0.0)
                jit_oderepr.rhs(states, 0.0, dy_jit)
            else:
                dy_eval = rhs(states, 0.0, parameters)
                jit_oderepr.rhs(states, 0.0, parameters, dy_jit)

            self.assertTrue(np.sum(np.abs((dy_eval-dy_correct))) < 1e-12)
            self.assertTrue(np.sum(np.abs((dy_jit-dy_correct))) < 1e-12)
            
            
    def test_matlab_python_code(self):
        from gotran.codegeneration.codegenerator import \
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
