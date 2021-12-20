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

__all__ = ["SymbolicNewtonSolution"]

from collections import OrderedDict
from functools import reduce

from modelparameters import sympy

# System imports
from modelparameters.sympytools import sp
from modelparameters.utils import check_arg, scalars

# Local imports
from gotran.model.ode import ODE


def _iszero(x):
    """Returns True if x is zero."""
    x = sp.sympify(x)
    return x.is_zero


class SymbolicNewtonSolution(object):
    """
    Class for storing information about a symbolic solution of newton
    iteration of an ODE
    """

    def __init__(self, ode, theta=1):
        """
        Create a SymbolicNewtonSolution

        Arguments
        ---------
        ode : ODE
            The ODE which the symbolc solution is created for
        theta : scalar \xe2\x88\x88 [0,1]
            Theta for creating the numerical integration rule of the ODE
        """

        check_arg(ode, ODE, 0, SymbolicNewtonSolution)
        check_arg(theta, scalars, 1, SymbolicNewtonSolution, ge=0, le=1)

        # Store attributes
        self.ode = ode
        self.theta = theta

        # Create symbolic linear system
        (
            self.F,
            self.F_expr,
            self.jacobi,
            self.states,
            self.jac_subs,
        ) = _create_newton_system(ode, theta)

        # Create a simplified LU decomposition of the jacobi matrix
        # FIXME: Better names!
        self.x, self.new_old, self.old_new = _LU_solve(self.jacobi, self.F)
        # self.LU_decomp, self.perm, self.LU_decomp_subs_0,\
        #    self.LU_decomp_subs_1 = _LU_solve(self.jacobi, self.F)


def _LU_solve(AA, rhs):
    """
    Returns the symbolic solve of AA*x=rhs
    """
    if not AA.is_square:
        raise sympy.NonSquareMatrixError()
    n = AA.rows
    A = AA[:, :]
    p = []

    nnz = 0
    for i in range(n):
        for j in range(n):
            nnz += not _iszero(A[i, j])

    print("Num non zeros in jacobian:", nnz)

    # A map between old symbols and new. The values in this dict corresponds to
    # where an old symbol is used. If the length of the value is 1 it is only
    # used once and once the value is used it can be freed.
    old_new = OrderedDict()
    new_old = OrderedDict()
    new_count = 0
    global zero_operations
    zero_operations = 0

    def update_entry(new_count, i, j, k):
        global zero_operations
        if _iszero(A[i, k] * A[k, j]):
            zero_operations += 1
            return new_count

        # Create new symbol and store the representation
        new_sym = sp.Symbol(f"j_{i}_{j}:{new_count}")
        new_old[new_sym] = A[i, j] - A[i, k] * A[k, j]
        for old_sym in [A[i, j], A[i, k], A[k, j]]:
            storage = old_new.get(old_sym)
            if storage is None:
                storage = set()
                old_new[old_sym] = storage
            storage.add(new_sym)

        # Change entry to the new symbol
        A[i, j] = new_sym
        new_count += 1

        return new_count

    # factorization
    for j in range(n):
        for i in range(j):
            for k in range(i):
                new_count = update_entry(new_count, i, j, k)
        pivot = -1
        for i in range(j, n):
            for k in range(j):
                new_count = update_entry(new_count, i, j, k)

            # find the first non-zero pivot, includes any expression
            if pivot == -1 and not _iszero(A[i, j]):
                pivot = i
        if pivot < 0:
            # this result is based on iszerofunc's analysis of the
            # possible pivots, so even though the element may not be
            # strictly zero, the supplied iszerofunc's evaluation gave
            # True
            raise ValueError("No nonzero pivot found; inversion failed.")

        if pivot != j:  # row must be swapped
            A.row_swap(pivot, j)
            p.append([pivot, j])
        scale = 1 / A[j, j]
        for i in range(j + 1, n):
            if _iszero(A[i, j]):
                zero_operations += 1
                continue

            # Create new symbol and store the representation
            new_sym = sp.Symbol(f"j_{i}_{j}:{new_count}")
            new_old[new_sym] = A[i, j] * scale
            for old_sym in [A[i, j], A[j, j]]:
                storage = old_new.get(old_sym)
                if storage is None:
                    storage = set()
                    old_new[old_sym] = storage
                storage.add(new_sym)

            # Change entry to the new symbol
            A[i, j] = new_sym
            new_count += 1

    nnz = 0
    for i in range(n):
        for j in range(n):
            nnz += not _iszero(A[i, j])

    print("Num non zeros in factorized jacobian:", nnz)
    print("Num non-zero operations while factorizing matrix:", new_count)
    print("Num zero operations while factorizing matrix:", zero_operations)
    factorizing_nnz_operations = new_count
    zero_operations = 0

    n = AA.rows
    b = rhs.permuteFwd(p)

    def update_entry(new_count, i, j):
        global zero_operations
        if _iszero(b[j, 0] * A[i, j]):
            zero_operations += 1
            return new_count

        # Create new symbol and store the representation
        new_sym = sp.Symbol(f"F_{i}:{new_count}")
        new_old[new_sym] = b[i, 0] - b[j, 0] * A[i, j]
        for old_sym in [b[i, 0], b[j, 0], A[i, j]]:
            storage = old_new.get(old_sym)
            if storage is None:
                storage = set()
                old_new[old_sym] = storage
            storage.add(new_sym)

        # Change entry to the new symbol
        b[i, 0] = new_sym
        new_count += 1

        return new_count

    # forward substitution, all diag entries are scaled to 1
    for i in range(n):
        for j in range(i):
            new_count = update_entry(new_count, i, j)
            # b.row(i, lambda x,k: x - b[j,k]*A[i,j])

    # backward substitution
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            new_count = update_entry(new_count, i, j)
            # b.row(i, lambda x,k: x - b[j,k]*A[i,j])
        b.row(i, lambda x, k: x / A[i, i])

    print(
        "Num zero operations while forward/backward substituting matrix:",
        zero_operations,
    )
    print(
        "Num non-zero operations while factorizing matrix:",
        new_count - factorizing_nnz_operations,
    )

    return b, new_old, old_new

    return A, p, old_new, new_old


def symbolic_solve(ode, theta=1):
    """
    Symbolically solve an ode system
    """


def _create_newton_system(ode, theta=1):
    """
    Return the nonlinear F, the Jacobian matrix and a mapping of states
    to nonero fields for the ODE
    """

    # Lists of symbols
    states = [state.sym for state in ode.states]
    states_0 = [state.sym_0 for state in ode.states]
    vars_ = [var.sym for var in ode.variables]
    vars_0 = [var.sym_0 for var in ode.variables]

    def sum_ders(ders):
        return reduce(lambda x, y: x + y, ders, 0)

    # Substitution dict between sym and previous sym value
    subs = [syms for syms in zip(states + vars_, states_0 + vars_0)]

    # Generate F using theta rule
    F_expr = [
        theta * expr
        + (1 - theta) * expr.subs(subs)
        - (sum_ders(ders) - sum_ders(ders).subs(subs)) / ode.dt
        for ders, expr in ode.get_derivative_expr(True)
    ]

    # Create mapping of orginal expression
    sym_map = OrderedDict()
    jac_subs = OrderedDict()
    F = []
    for i, expr in enumerate(F_expr):
        for j, state in enumerate(states):
            F_ij = expr.diff(state)
            if F_ij:
                jac_sym = sp.Symbol(f"j_{i}_{j}")
                sym_map[i, j] = jac_sym
                jac_subs[jac_sym] = F_ij
                # print "[%d,%d] (%d) # [%s, %s]  \n%s" \
                # % (i,j, len(F_ij.args), states[i], states[j], F_ij)

        F_i = sp.Symbol(f"F_{i}")
        sym_map[i, j + 1] = F_i
        jac_subs[F_i] = expr
        F.append(F_i)

    # Create the Jacobian
    jacobi = sp.SparseMatrix(
        len(states),
        len(states),
        lambda i, j: sym_map.get((i, j), 0),
    )

    # return Symbolic representation of the linear system
    return sp.Matrix(F), F_expr, jacobi, states, jac_subs
