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

# ModelParameter imports
from modelparameters.parameters import Param, OptionParam, ScalarParam
from modelparameters.parameterdict import ParameterDict

parameters = ParameterDict(
    code_generation = ParameterDict(
        array = ParameterDict(
            index_format=OptionParam("[]", ["[]", "{}", "()"],
                               description="The format of index notations."),
            index_offset=ScalarParam(0, ge=0,
                               description="A global offset to all indexed variables."),
            flatten=Param(False,
                          description="If true multidimensional arrays will be "\
                          "flattened. jac[2,3] -> jac[27] if the shape of jac "\
                          "is (12,12)")
            ),
        
        parameters = ParameterDict(
            representation = OptionParam("named", ["named", "array", "numerals"],
                                         description="Controls how parameters are "\
                                         "represented in the code. As named variables,"\
                                         " as an indexed array or as the default "\
                                         "numeral values given in the gotran model."),
            array_name = Param("parameters", description="The name of the array "\
                               "representing the parameters."),
            ),
        
        states = ParameterDict(
            representation = OptionParam("named", ["named", "array"],
                                         description="Controls how states are "\
                                         "represented in the code. As named variables,"\
                                         " or as an indexed array."),
            array_name = Param("states", description="The name of the array "\
                               "representing the states."),
            ),
        
        body = ParameterDict(
            representation = OptionParam("named", ["named", "array", "reused_array"],
                                         description="Controls how body variables are "\
                                         "represented in the code. As named variables,"\
                                         "as an indexed array or as indexed array with "\
                                         "reuse of unused array elements."),
            
            array_name = Param("body", description="The name of the array "\
                               "representing the body."),
            
            optimize_exprs = OptionParam("none", ["none", "numerals",
                                                  "numerals_symbols"],
                                         description="Remove body expressions as "\
                                         "intermediates, which contains only a "\
                                         "numeral or numerals and a symbol."),
            
            ),
        ),
    )
