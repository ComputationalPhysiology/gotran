# Copyright (C) 2013-2014 Johan Hake
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

__all__ = ["parameters"]

from modelparameters.parameterdict import ParameterDict

# ModelParameter imports
from modelparameters.parameters import OptionParam, Param, ScalarParam

parameters = ParameterDict(
    # Generation parameters
    generation=ParameterDict(
        class_code=Param(
            False,
            description="If true methods are contained " "inside a class",
        ),
        # Code generation parameters
        code=ParameterDict(
            # Float precision
            float_precision=OptionParam(
                "double",
                ["double", "single"],
                description="Float precision in generated " "code.",
            ),
            # Parameter for default argument order
            default_arguments=OptionParam(
                "stp",
                ["tsp", "stp", "spt", "ts", "st"],
                description="Default input argument order: "
                "s=states, p=parameters, t=time",
            ),
            # Parameter for the time parameter name
            time=ParameterDict(name=Param("t", description="Name of time argument")),
            # Parameter for the time step parameter name
            dt=ParameterDict(
                name=Param("dt", description="Name of time step argument"),
            ),
            # Number of nodes for solvers
            n_nodes=ScalarParam(0, ge=0, description="Number of nodes for simulation"),
            # Parameters for code generation of arrays
            array=ParameterDict(
                index_format=OptionParam(
                    "[]",
                    ["[]", "{}", "()"],
                    description="The format of index notations.",
                ),
                index_offset=ScalarParam(
                    0,
                    ge=0,
                    description="A global offset to all indexed variables.",
                ),
                flatten=Param(
                    True,
                    description="If true multidimensional arrays will be "
                    "flattened. jac[2,3] -> jac[27] if the shape of jac "
                    "is (12,12)",
                ),
            ),
            # Parameters for code generation of parameters
            parameters=ParameterDict(
                field_parameters=Param(
                    [""],
                    description="A list of parameter names "
                    "which should be treated as field parameters. "
                    "Only available for CUDA backends.",
                ),
                representation=OptionParam(
                    "named",
                    ["named", "array", "numerals"],
                    description="Controls how parameters are "
                    "represented in the code. As named variables,"
                    " as an indexed array or as the default "
                    "numeral values given in the gotran model.",
                ),
                array_name=Param(
                    "parameters",
                    description="The name of the array " "representing the parameters.",
                ),
                field_array_name=Param(
                    "field_parameters",
                    description="The name of the array "
                    "representing the field parameters.",
                ),
                add_offset=Param(
                    False,
                    description="If true an offset will be "
                    "added to the index of each parameter",
                ),
                add_field_offset=Param(
                    False,
                    description="If true an offset will be "
                    "added to the index of each field parameter",
                ),
            ),
            # Parameters for code generation of states
            states=ParameterDict(
                field_states=Param(
                    [""],
                    description="A list of state names "
                    "which should be treated as field states. "
                    "Not available for all backends...",
                ),
                representation=OptionParam(
                    "named",
                    ["named", "array"],
                    description="Controls how states are "
                    "represented in the code. As named variables,"
                    " or as an indexed array.",
                ),
                array_name=Param(
                    "states",
                    description="The name of the array " "representing the states.",
                ),
                add_offset=Param(
                    False,
                    description="If true an offset will be "
                    "added to the index of each state",
                ),
            ),
            # Parameters for code generation of body expressions
            body=ParameterDict(
                use_cse=Param(
                    False,
                    description="If true will the body be "
                    "optimized using SymPy common sub expression "
                    "extraction.",
                ),
                use_enum=Param(
                    False,
                    description="If true use enumeration"
                    "data types instead of indexing.",
                ),
                in_signature=Param(
                    False,
                    description="If true the body argument "
                    "will be included in the signature.",
                ),
                representation=OptionParam(
                    "named",
                    ["named", "array", "reused_array"],
                    description="Controls how body variables are "
                    "represented in the code. As named variables,"
                    "as an indexed array or as indexed array with "
                    "reuse of unused array elements.",
                ),
                array_name=Param(
                    "body",
                    description="The name of the array " "representing the body.",
                ),
                optimize_exprs=OptionParam(
                    "none",
                    ["none", "numerals", "numerals_symbols"],
                    description="Remove body expressions as "
                    "intermediates, which contains only a "
                    "numeral or numerals and a symbol.",
                ),
            ),
        ),
        # Parameters for automatic generation of specific functions
        functions=ParameterDict(
            rhs=ParameterDict(
                generate=Param(
                    True,
                    description="Generate code for the "
                    "evaluation of the right hand side evaluation.",
                ),
                function_name=Param(
                    "rhs",
                    description="The name of " "the generated function.",
                ),
                result_name=Param(
                    "values",
                    description="The name of " "the result argument.",
                ),
            ),
            monitored=ParameterDict(
                generate=Param(
                    True,
                    description="Generate code for the "
                    "evaluation of monitored intermediates.",
                ),
                function_name=Param(
                    "monitor",
                    description="The name of " "the generated function.",
                ),
                result_name=Param(
                    "monitored",
                    description="The name of " "the result argument.",
                ),
            ),
            jacobian=ParameterDict(
                generate=Param(
                    False,
                    description="Generate code for the "
                    "evaluation of the jacobian of the right hand "
                    "side.",
                ),
                function_name=Param(
                    "compute_jacobian",
                    description="The name " "of the generated function.",
                ),
                result_name=Param(
                    "jac",
                    description="The name of " "the result argument.",
                ),
            ),
            lu_factorization=ParameterDict(
                generate=Param(
                    False,
                    description="Generate code for "
                    "symbolicly factorize the jacobian.",
                ),
                function_name=Param(
                    "lu_factorize",
                    description="The name " "of the generated function.",
                ),
            ),
            forward_backward_subst=ParameterDict(
                generate=Param(
                    False,
                    description="Generate code for the "
                    "symbolic forward backward substitution of the "
                    "jacobian.",
                ),
                function_name=Param(
                    "forward_backward_subst",
                    description="The name of the generated " "function.",
                ),
                residual_name=Param(
                    "F",
                    description="The name of " "the residual argument.",
                ),
                result_name=Param(
                    "dx",
                    description="The name of " "the incriment argument.",
                ),
            ),
            componentwise_rhs_evaluation=ParameterDict(
                generate=Param(
                    False,
                    description="If true, generate code for "
                    "computing componentwise evaluation of the rhs.",
                ),
                function_name=Param(
                    "componentwise_rhs",
                    description="The name of the generated " "function.",
                ),
            ),
            linearized_rhs_evaluation=ParameterDict(
                generate=Param(
                    False,
                    description="If true, generate code for "
                    "computing linearized evaluation of linear rhs "
                    "terms.",
                ),
                function_name=Param(
                    "linearized_rhs",
                    description="The name of the generated " "function.",
                ),
                include_rhs=Param(
                    False,
                    description="If True the rhs will be "
                    "included as a result argument.",
                ),
                only_linear=Param(
                    True,
                    description="If True only linearized "
                    "expressions for the linear derivatives will "
                    "be generated.",
                ),
                result_names=Param(
                    ["linearized", "rhs"],
                    description="The name of " "the two results arguments.",
                ),
            ),
        ),
        # Parameters for automatic generation of lists/arrays of names for states and parameters
        lists=ParameterDict(
            parameter_names=ParameterDict(
                generate=Param(False, description="Generate list of parameter names"),
                name=Param("parameter_names", description="Name of list variable"),
            ),
            state_names=ParameterDict(
                generate=Param(False, description="Generate list of state names"),
                name=Param("state_names", description="Name of list variable"),
            ),
        ),
        # Parameters for automatic generation of specific solver functions
        solvers=ParameterDict(
            explicit_euler=ParameterDict(
                generate=Param(
                    False,
                    description="If true, generate code for "
                    "solving an ODE using explicit Euler method.",
                ),
                function_name=Param(
                    "forward_explicit_euler",
                    description="The name of the generated " "function.",
                ),
            ),
            rush_larsen=ParameterDict(
                generate=Param(
                    False,
                    description="If true, generate code for "
                    "solving an ODE using Rush Larsen method.",
                ),
                function_name=Param(
                    "forward_rush_larsen",
                    description="The name of the generated " "function.",
                ),
                delta=ScalarParam(
                    1e-8,
                    gt=0,
                    lt=1.0,
                    description="Value to "
                    "safeguard the evaluation of the rush larsen "
                    "step.",
                ),
            ),
            generalized_rush_larsen=ParameterDict(
                generate=Param(
                    False,
                    description="If true, generate code for "
                    "solving an ODE using generalized Rush Larsen method.",
                ),
                function_name=Param(
                    "forward_generalized_rush_larsen",
                    description="The name of the generated " "function.",
                ),
                delta=ScalarParam(
                    1e-8,
                    gt=0,
                    lt=1.0,
                    description="Value to "
                    "safeguard the evaluation of the rush larsen "
                    "step.",
                ),
            ),
            hybrid_generalized_rush_larsen=ParameterDict(
                generate=Param(
                    False,
                    description="If true, generate code for "
                    "solving an ODE using a hybrid generalized Rush Larsen method.",
                ),
                function_name=Param(
                    "forward_hybrid_generalized_rush_larsen",
                    description="The name of the generated " "function.",
                ),
                delta=ScalarParam(
                    1e-8,
                    gt=0,
                    lt=1.0,
                    description="Value to "
                    "safeguard the evaluation of the rush larsen "
                    "step.",
                ),
                stiff_states=Param(
                    "",
                    name="stiff_states",
                    description="List of names of state variables that should "
                    "be solved with the GRL1 scheme that has better stability"
                    "than explicit Euler",
                ),
            ),
            simplified_implicit_euler=ParameterDict(
                generate=Param(
                    False,
                    description="If true, generate code for "
                    "solving an ODE using Rush Larsen method.",
                ),
                function_name=Param(
                    "forward_simplified_implicit_euler",
                    description="The name of the generated " "function.",
                ),
                numeric_jacobian=Param(
                    False,
                    description="Use numeric " "calculated diagonal jacobian.",
                ),
            ),
        ),
    ),
    # Parameters for different input
    input=ParameterDict(
        # Parameters for CellML input
        cellml=ParameterDict(
            change_state_names=Param(
                [],
                description="A list of state names "
                "which should be changed to not interfere "
                "with for example a parameter name.",
            ),
            grouping=OptionParam(
                "encapsulation",
                ["encapsulation", "containment"],
                description="Determines what type of grouping "
                "should be used when the cellml model is parsed.",
            ),
            use_sympy_integers=Param(
                False,
                description="If yes dedicated sympy "
                "integers will be used instead of the "
                "integers 0-10. This will turn 1/2 into a "
                "sympy rational instead of the float 0.5.",
            ),
            strip_parent_name=Param(
                True,
                description="If True strip the name from "
                "the child component it contains the name of the "
                "parent component.",
            ),
        ),
    ),
)
