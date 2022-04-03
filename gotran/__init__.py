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

__version__ = "2022.1.1"


# import algorithms
# Import gotran modules
from . import codegeneration, common, input, model

# from algorithms import *
from .codegeneration import (
    CCodeGenerator,
    CodeComponent,
    CommonSubExpressionODE,
    CppCodeGenerator,
    CUDACodeGenerator,
    FactorizedJacobianComponent,
    ForwardBackwardSubstitutionComponent,
    JacobianActionComponent,
    JacobianComponent,
    JuliaCodeGenerator,
    LatexCodeGenerator,
    LinearizedDerivativeComponent,
    MatlabCodeGenerator,
    PythonCodeGenerator,
    algorithmcomponents,
    class_name,
    codecomponent,
    codegenerators,
    componentwise_derivative,
    diagonal_jacobian_action_expressions,
    diagonal_jacobian_expressions,
    factorized_jacobian_expressions,
    forward_backward_subst_expressions,
    jacobian_action_expressions,
    jacobian_expressions,
    latexcodegenerator,
    linearized_derivatives,
    monitored_expressions,
    rhs_expressions,
    solvercomponents,
)

# Import classes and routines from gotran modules
from .common import GotranException, parameters
from .input import CellMLParser, cellml, cellml2ode
from .model import (
    ODE,
    AlgebraicExpression,
    Comment,
    DerivativeExpression,
    Derivatives,
    Dt,
    Expression,
    IndexedExpression,
    IndexedObject,
    Intermediate,
    ODEComponent,
    ODEObject,
    ODEValueObject,
    Parameter,
    RateExpression,
    SingleODEObjects,
    State,
    StateDerivative,
    StateExpression,
    StateSolution,
    Time,
    cmp,
    cmp_to_key,
    exec_ode,
    expressions,
    load_ode,
    loadmodel,
    ode,
    odecomponent,
    odeobjects,
    recreate_expression,
)

from modelparameters.logger import set_log_level


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
    "odeobjects",
    "ODEObject",
    "Comment",
    "ODEValueObject",
    "Parameter",
    "State",
    "SingleODEObjects",
    "Time",
    "Dt",
    "IndexedObject",
    "cmp_to_key",
    "cmp",
    "odecomponent",
    "ODEComponent",
    "expressions",
    "Expression",
    "DerivativeExpression",
    "AlgebraicExpression",
    "StateExpression",
    "StateSolution",
    "RateExpression",
    "Intermediate",
    "StateDerivative",
    "Derivatives",
    "IndexedExpression",
    "recreate_expression",
    "ode",
    "ODE",
    "loadmodel",
    "load_ode",
    "exec_ode",
    "GotranException",
    "parameters",
    "common",
    "model",
    "input",
    "codegeneration",
    "cellml",
    "cellml2ode",
    "CellMLParser",
    "set_log_level",
]
