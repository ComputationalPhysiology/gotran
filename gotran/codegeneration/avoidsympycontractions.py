# Copyright (C) 2013 Johan Hake
#
# This file is part of Gotran
#
# ModelParameters is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelParameters is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Gotran. If not, see <http://www.gnu.org/licenses/>.
# Not meant to make any of the functions available from this module
__all__ = []

import mpmath.libmp as _mlib

# A hack to get around evaluation of SymPy expressions
import modelparameters.sympy as sp
from modelparameters.sympy.core import function as _function
from modelparameters.sympy.core.expr import Expr as _Expr

_evaluate = False


def enable_evaluation():
    """
    Enable Add, Mul and Pow contractions
    """
    global _evaluate
    _evaluate = True


def disable_evaluation():
    """
    Disable Add, Mul and Pow contractions
    """
    global _evaluate
    _evaluate = False


def _assocop_new(cls, *args, **options):
    args = list(map(sp.sympify, args))
    args = [a for a in args if a is not cls.identity]

    if not options.pop("evaluate", _evaluate):
        return cls._from_args(args)

    if len(args) == 0:
        return cls.identity
    if len(args) == 1:
        return args[0]

    c_part, nc_part, order_symbols = cls.flatten(args)
    is_commutative = not nc_part
    obj = cls._from_args(c_part + nc_part, is_commutative)

    if order_symbols is not None:
        # Where does this come from?
        return C.Order(obj, *order_symbols)  # noqa: F821
    return obj


def _function_new(cls, *args, **options):
    # Handle calls like Function('f')
    if cls is _function.Function:
        return _function.UndefinedFunction(*args)

    if cls.nargs is not None:
        if isinstance(cls.nargs, tuple):
            nargs = cls.nargs
        else:
            nargs = (cls.nargs,)

        n = len(args)

        if n not in nargs:
            # XXX: exception message must be in exactly this format to make
            # it work with NumPy's functions like vectorize(). The ideal
            # solution would be just to attach metadata to the exception
            # and change NumPy to take advantage of this.
            temp = (
                "%(name)s takes exactly %(args)s "
                "argument%(plural)s (%(given)s given)"
            )
            raise TypeError(
                temp
                % {
                    "name": cls,
                    "args": cls.nargs,
                    "plural": "s" * (n != 1),
                    "given": n,
                },
            )

    evaluate = options.get("evaluate", _evaluate)
    result = super(_function.Function, cls).__new__(cls, *args, **options)
    if not evaluate or not isinstance(result, cls):
        return result

    pr = max(cls._should_evalf(a) for a in result.args)
    pr2 = min(cls._should_evalf(a) for a in result.args)
    if pr2 > 0:
        return result.evalf(_mlib.libmpf.prec_to_dps(pr))
    return result


def _pow_new(cls, b, e, evaluate=True):
    # don't optimize "if e==0; return 1" here; it's better to handle that
    # in the calling routine so this doesn't get called
    b = sp.sympify(b)
    e = sp.sympify(e)
    if _evaluate and evaluate:
        if e is sp.S.Zero:
            return sp.S.One
        elif e is sp.S.One:
            return b
        elif sp.S.NaN in (b, e):
            if b is sp.S.One:  # already handled e == 0 above
                return sp.S.One
            return sp.S.NaN
        else:
            obj = b._eval_power(e)
            if obj is not None:
                return obj

    obj = _Expr.__new__(cls, b, e)
    obj.is_commutative = b.is_commutative and e.is_commutative
    return obj


# Overload new method with none evaluating one
# FIXME: Need to look at inheritance
# _AssocOp.__new__ = types.MethodType(_cacheit(_assocop_new), None, _ManagedProperties)
# _Pow.__new__ = types.MethodType(_cacheit(_pow_new), None, _ManagedProperties)
# _function.Function.__new__ = types.MethodType(_cacheit(_function_new), None, _ManagedProperties)
