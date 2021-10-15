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
from . import (
    algorithmcomponents,
    codecomponent,
    codegenerators,
    compilemodule,
    latexcodegenerator,
    oderepresentation,
    solvercomponents,
)
from .algorithmcomponents import (
    CommonSubExpressionODE,
    FactorizedJacobianComponent,
    ForwardBackwardSubstitutionComponent,
    JacobianActionComponent,
    JacobianComponent,
    LinearizedDerivativeComponent,
    componentwise_derivative,
    diagonal_jacobian_action_expressions,
    diagonal_jacobian_expressions,
    factorized_jacobian_expressions,
    forward_backward_subst_expressions,
    jacobian_action_expressions,
    jacobian_expressions,
    linearized_derivatives,
    monitored_expressions,
    rhs_expressions,
)

# Import classes and routines from gotran modules
from .codecomponent import CodeComponent
from .codegenerators import (
    CCodeGenerator,
    CppCodeGenerator,
    CUDACodeGenerator,
    JuliaCodeGenerator,
    MatlabCodeGenerator,
    PythonCodeGenerator,
    class_name,
)
from .compilemodule import compile_module
from .latexcodegenerator import LatexCodeGenerator
from .solvercomponents import (
    ExplicitEuler,
    GeneralizedRushLarsen,
    HybridGeneralizedRushLarsen,
    RushLarsen,
    SimplifiedImplicitEuler,
    explicit_euler_solver,
    generalized_rush_larsen_solver,
    get_solver_fn,
    hybrid_generalized_rush_larsen_solver,
    rush_larsen_solver,
    simplified_implicit_euler_solver,
)

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
    "oderepresentation",
]
