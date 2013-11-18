# Copyright (C) 2013 Johan Hake
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

__all__ = ["ode_primitives"]

from sympy.core.function import AppliedUndef
from sympy import preorder_traversal, Symbol

def ode_primitives(expr, time):
    """
    Return all ODE primitives

    Arguments:
    ----------
    expr : sympy expressio
        A sympy expression of ode symbols
    time : sympy Symbol
        A Symbol representing time in the ODE
    """
    symbols = set()
    pt = preorder_traversal(expr)
    
    for node in pt:
        
        # Collect AppliedUndefs which are functions of time
        if isinstance(node, AppliedUndef) and len(node.args) == 1 and \
               node.args[0] == time:
            pt.skip()
            symbols.add(node)
        elif(isinstance(node, Symbol)):
            symbols.add(node)

    return symbols

