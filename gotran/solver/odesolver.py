# Global imports
import numpy as np


# Local imports
from ._assimulo import *
from ._scipy import *
from .utils import *


# class ODESolver(object):
   
    # def __init__(self, ode, method="cvode", **options):
def ODESolver(ode, method="scipy", **options):
     """
    A generic ODE solver for solving problem of the types on the form,
        
    .. math::
    
        \dot{y} = f(t,y), \quad y(t_0) = y_0.
            
    Here one need to specific the backend which is either Scipy or Assimulo.
    
    *Arguments*

    ode : gotran.ODE or gotran.CellModel
        The ode you want to solve in a gotran object
    method : str
       Solver method. Possible inputs are  or 'scipy' (Default:'sundials')

    options : dict:
       Options for the solver, see `list_solver_options`
    """
     
     check_method(method)

        
    
     if method == "scipy":
         return ScipySolver(ode, **options)
     elif method in sundials_methods:
         return AssimuloSolver(ode, method, **options)
     elif method in gotran_methods:
         raise NotImplementedError
     elif method in goss_methods:
         raise NotImplementedError
     else:
         raise NotImplementedError
        

    

    # @staticmethod
    # def list_solver_method():
    #     return methods
    
              
    # def get_solver_options(self):
    #     """
    #     Get solver options for the current solver
    #     """

    #     return self.odesolver.get_options()
        


    # @property
    # def odesolver(self):
    #     return self._solver

    # def solve(self, *args, **kwargs):


        
    #     ts, ys = self.odesolver.solve(*args, **kwargs)

    #     return ts, ys
        



