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

__version__ = "2021.0.2"


# Import gotran modules
from . import common
from . import model

# import algorithms
from . import codegeneration
from . import input

# from . import scripts

# Import classes and routines from gotran modules
from .common import (
    GotranException,
    parameters,
)

from .model import (
    odeobjects,
    ODEObject,
    Comment,
    ODEValueObject,
    Parameter,
    State,
    SingleODEObjects,
    Time,
    Dt,
    IndexedObject,
    cmp_to_key,
    cmp,
    odecomponent,
    ODEComponent,
    expressions,
    Expression,
    DerivativeExpression,
    AlgebraicExpression,
    StateExpression,
    StateSolution,
    RateExpression,
    Intermediate,
    StateDerivative,
    Derivatives,
    IndexedExpression,
    recreate_expression,
    ode,
    ODE,
    loadmodel,
    load_ode,
    exec_ode,
)


# from algorithms import *
from .codegeneration import (
    codecomponent,
    CodeComponent,
    algorithmcomponents,
    JacobianComponent,
    JacobianActionComponent,
    FactorizedJacobianComponent,
    ForwardBackwardSubstitutionComponent,
    LinearizedDerivativeComponent,
    CommonSubExpressionODE,
    componentwise_derivative,
    linearized_derivatives,
    jacobian_expressions,
    jacobian_action_expressions,
    factorized_jacobian_expressions,
    forward_backward_subst_expressions,
    diagonal_jacobian_expressions,
    rhs_expressions,
    diagonal_jacobian_action_expressions,
    monitored_expressions,
    solvercomponents,
    JacobianComponent,
    JacobianActionComponent,
    FactorizedJacobianComponent,
    ForwardBackwardSubstitutionComponent,
    LinearizedDerivativeComponent,
    CommonSubExpressionODE,
    componentwise_derivative,
    linearized_derivatives,
    jacobian_expressions,
    jacobian_action_expressions,
    factorized_jacobian_expressions,
    forward_backward_subst_expressions,
    diagonal_jacobian_expressions,
    rhs_expressions,
    diagonal_jacobian_action_expressions,
    monitored_expressions,
    codegenerators,
    PythonCodeGenerator,
    CCodeGenerator,
    CppCodeGenerator,
    MatlabCodeGenerator,
    class_name,
    CUDACodeGenerator,
    JuliaCodeGenerator,
    latexcodegenerator,
    LatexCodeGenerator,
)
from .input import cellml, cellml2ode, CellMLParser

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
]
