# Copyright (C) 2012 Johan Hake
#
# This file is part of Gotran.
#
# Gotran is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Gotran is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Gotran. If not, see <http://www.gnu.org/licenses/>.
# Import gotran modules
from . import algorithmcomponents
from . import codecomponent
from . import codegenerators
from . import compilemodule
from . import latexcodegenerator
from . import oderepresentation
from . import solvercomponents
from .algorithmcomponents import CommonSubExpressionODE
from .algorithmcomponents import componentwise_derivative
from .algorithmcomponents import diagonal_jacobian_action_expressions
from .algorithmcomponents import diagonal_jacobian_expressions
from .algorithmcomponents import factorized_jacobian_expressions
from .algorithmcomponents import FactorizedJacobianComponent
from .algorithmcomponents import forward_backward_subst_expressions
from .algorithmcomponents import ForwardBackwardSubstitutionComponent
from .algorithmcomponents import jacobian_action_expressions
from .algorithmcomponents import jacobian_expressions
from .algorithmcomponents import JacobianActionComponent
from .algorithmcomponents import JacobianComponent
from .algorithmcomponents import linearized_derivatives
from .algorithmcomponents import LinearizedDerivativeComponent
from .algorithmcomponents import monitored_expressions
from .algorithmcomponents import rhs_expressions
from .codecomponent import CodeComponent
from .codegenerators import CCodeGenerator
from .codegenerators import class_name
from .codegenerators import CppCodeGenerator
from .codegenerators import CUDACodeGenerator
from .codegenerators import JuliaCodeGenerator
from .codegenerators import MatlabCodeGenerator
from .codegenerators import PythonCodeGenerator
from .compilemodule import compile_module
from .compilemodule import has_cppyy
from .latexcodegenerator import LatexCodeGenerator
from .solvercomponents import explicit_euler_solver
from .solvercomponents import ExplicitEuler
from .solvercomponents import generalized_rush_larsen_solver
from .solvercomponents import GeneralizedRushLarsen
from .solvercomponents import get_solver_fn
from .solvercomponents import hybrid_generalized_rush_larsen_solver
from .solvercomponents import HybridGeneralizedRushLarsen
from .solvercomponents import rush_larsen_solver
from .solvercomponents import RushLarsen
from .solvercomponents import simplified_implicit_euler_solver
from .solvercomponents import SimplifiedImplicitEuler

# Import classes and routines from gotran modules

# Assign the __all__ attribute
__all__ = [
    "codecomponent",
    "CodeComponent",
    "algorithmcomponents",
    "JacobianComponent",
    "JacobianActionComponent",
    "FactorizedJacobianComponent",
    "ForwardBackwardSubstitutionComponent",
    "LinearizedDerivativeComponent",
    "CommonSubExpressionODE",
    "componentwise_derivative",
    "linearized_derivatives",
    "jacobian_expressions",
    "jacobian_action_expressions",
    "factorized_jacobian_expressions",
    "forward_backward_subst_expressions",
    "diagonal_jacobian_expressions",
    "rhs_expressions",
    "diagonal_jacobian_action_expressions",
    "monitored_expressions",
    "solvercomponents",
    "JacobianComponent",
    "JacobianActionComponent",
    "FactorizedJacobianComponent",
    "ForwardBackwardSubstitutionComponent",
    "LinearizedDerivativeComponent",
    "CommonSubExpressionODE",
    "componentwise_derivative",
    "linearized_derivatives",
    "jacobian_expressions",
    "jacobian_action_expressions",
    "factorized_jacobian_expressions",
    "forward_backward_subst_expressions",
    "diagonal_jacobian_expressions",
    "rhs_expressions",
    "diagonal_jacobian_action_expressions",
    "monitored_expressions",
    "codegenerators",
    "PythonCodeGenerator",
    "CCodeGenerator",
    "CppCodeGenerator",
    "MatlabCodeGenerator",
    "class_name",
    "CUDACodeGenerator",
    "JuliaCodeGenerator",
    "latexcodegenerator",
    "LatexCodeGenerator",
    "compile_module",
    "ExplicitEuler",
    "explicit_euler_solver",
    "RushLarsen",
    "rush_larsen_solver",
    "GeneralizedRushLarsen",
    "generalized_rush_larsen_solver",
    "HybridGeneralizedRushLarsen",
    "hybrid_generalized_rush_larsen_solver",
    "SimplifiedImplicitEuler",
    "simplified_implicit_euler_solver",
    "get_solver_fn",
    "compilemodule",
    "has_cppyy",
    "oderepresentation",
]
