# Global imports
import numpy as np


# Local imports
from ._assimulo import *
from ._scipy import *
from .utils import *


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
          try:
               return AssimuloSolver(ode, method, **options)
          except:
               print("Could not import Sundials solvers. Use Scipy ODE solver instead")
               return ScipySolver(ode)
               
     elif method in gotran_methods:
         raise NotImplementedError
     elif method in goss_methods:
         raise NotImplementedError
     else:
         raise NotImplementedError
        


