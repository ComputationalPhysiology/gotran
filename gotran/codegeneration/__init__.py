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
from . import codegenerators
from . import compilemodule
from . import latexcodegenerator
from . import codecomponent
from . import algorithmcomponents
from . import oderepresentation
from . import solvercomponents

# Import classes and routines from gotran modules
from .codecomponent import CodeComponent
from .algorithmcomponents import (JacobianComponent, JacobianActionComponent,
                                  FactorizedJacobianComponent,
                                  ForwardBackwardSubstitutionComponent,
                                  LinearizedDerivativeComponent,
                                  CommonSubExpressionODE,
                                  componentwise_derivative,
                                  linearized_derivatives, jacobian_expressions,
                                  jacobian_action_expressions,
                                  factorized_jacobian_expressions,
                                  forward_backward_subst_expressions,
                                  diagonal_jacobian_expressions,
                                  rhs_expressions,
                                  diagonal_jacobian_action_expressions,
                                  monitored_expressions)
from .solvercomponents import (ExplicitEuler, explicit_euler_solver,
                               RushLarsen, rush_larsen_solver,
                               GeneralizedRushLarsen,
                               generalized_rush_larsen_solver,
                               SimplifiedImplicitEuler,
                               simplified_implicit_euler_solver,
                               get_solver_fn)
from .codegenerators import (PythonCodeGenerator, CCodeGenerator,
                             CppCodeGenerator, MatlabCodeGenerator, class_name,
                             CUDACodeGenerator, JuliaCodeGenerator)
from .latexcodegenerator import LatexCodeGenerator, _default_latex_params
from .compilemodule import compile_module
# from .oderepresentation import *


# Assign the __all__ attribute
# __all__ = [name for name in list(globals().keys()) if name[:1] != "_"]
__all__ = ["codecomponent", "CodeComponent", "algorithmcomponents",
           "JacobianComponent", "JacobianActionComponent",
           "FactorizedJacobianComponent",
           "ForwardBackwardSubstitutionComponent",
           "LinearizedDerivativeComponent",
           "CommonSubExpressionODE", "componentwise_derivative",
           "linearized_derivatives", "jacobian_expressions",
           "jacobian_action_expressions", "factorized_jacobian_expressions",
           "forward_backward_subst_expressions",
           "diagonal_jacobian_expressions", "rhs_expressions",
           "diagonal_jacobian_action_expressions",
           "monitored_expressions", "solvercomponents", "JacobianComponent",
           "JacobianActionComponent",
           "FactorizedJacobianComponent",
           "ForwardBackwardSubstitutionComponent",
           "LinearizedDerivativeComponent",
           "CommonSubExpressionODE", "componentwise_derivative",
           "linearized_derivatives", "jacobian_expressions",
           "jacobian_action_expressions", "factorized_jacobian_expressions",
           "forward_backward_subst_expressions",
           "diagonal_jacobian_expressions", "rhs_expressions",
           "diagonal_jacobian_action_expressions",
           "monitored_expressions", "codegenerators", "PythonCodeGenerator",
           "CCodeGenerator", "CppCodeGenerator", "MatlabCodeGenerator",
           "class_name", "CUDACodeGenerator", "JuliaCodeGenerator",
           "latexcodegenerator", "LatexCodeGenerator"]
