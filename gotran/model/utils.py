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

__all__ = [
    "ode_primitives",
    "INTERMEDIATE",
    "ALGEBRAIC_EXPRESSION",
    "DERIVATIVE_EXPRESSION",
    "STATE_SOLUTION_EXPRESSION",
    "special_expression",
    "iter_objects",
    "ode_objects",
    "ode_components",
    "ODEObjectList",
    "RateDict",
]

import re
import weakref

# System imports
from collections import OrderedDict

from modelparameters.logger import error

# ModelParameters imports
from modelparameters.sympytools import sp
from modelparameters.utils import check_arg, tuplewrap
from modelparameters.sympy import Symbol, preorder_traversal
from modelparameters.sympy.core.function import AppliedUndef

from .expressions import State

# Local imports
from .odeobjects import ODEObject


def ode_primitives(expr, time):
    """
    Return all ODE primitives

    Arguments
    ---------
    expr : sympy.expression
        A sympy expression of ode symbols
    time : sympy.Symbol
        A Symbol representing time in the ODE
    """
    symbols = set()
    pt = preorder_traversal(expr)

    for node in pt:

        # Collect AppliedUndefs which are functions of time
        if (
            isinstance(node, AppliedUndef)
            and len(node.args) == 1
            and node.args[0] == time
        ):
            pt.skip()
            symbols.add(node)
        elif isinstance(node, Symbol):
            symbols.add(node)

    return symbols


_derivative_name_template = re.compile(r"\Ad([a-zA-Z]\w*)_d([a-zA-Z]\w*)\Z")
_algebraic_name_template = re.compile(r"\Aalg_([a-zA-Z]\w*)_0\Z")

# Flags for determine special expressions
INTERMEDIATE = 0
ALGEBRAIC_EXPRESSION = 1
DERIVATIVE_EXPRESSION = 2
STATE_SOLUTION_EXPRESSION = 3


def special_expression(name, root):
    """
    Check if an expression name corresponds to a special expression
    """

    alg_expr = re.search(_algebraic_name_template, name)
    if alg_expr:
        return alg_expr, ALGEBRAIC_EXPRESSION

    der_expr = re.search(_derivative_name_template, name)
    if der_expr:
        return der_expr, DERIVATIVE_EXPRESSION

    state = root.present_ode_objects.get(name)
    if state and isinstance(state, State):
        return state, STATE_SOLUTION_EXPRESSION

    return None, INTERMEDIATE


class iter_objects(object):
    """
    A recursive iterator over all objects of a component including its
    childrens

    Arguments
    ---------
    comp : gotran.ODEComponent
        The root ODEComponent of the iteration
    reverse : bool
        If True the iteration is done from the last component added
    types : gotran.ODEObject types (optional)
        Only iterate over particular types

    Yields
    ------
    ode_object : gotran.ODEObject
        All ODEObjects of a component
    """

    def __init__(
        self, comp, return_comp=True, only_return_comp=False, reverse=False, *types
    ):
        from .odecomponent import ODEComponent

        assert isinstance(comp, ODEComponent)
        self._types = tuplewrap(types) or (ODEObject,)
        self._return_comp = return_comp
        self._only_return_comp = only_return_comp
        if reverse:
            self._object_iterator = self._reverse_iter_objects(comp)
        else:
            self._object_iterator = self._iter_objects(comp)

        assert all(issubclass(T, ODEObject) for T in self._types)

    def _reverse_iter_objects(self, comp):

        # First all children components in reversed order
        for sub_comp in reversed(list(comp.children.values())):
            for sub_tree in self._reverse_iter_objects(sub_comp):
                yield sub_tree

        # Secondly return component
        if self._return_comp or self._only_return_comp:
            yield comp

        if self._only_return_comp:
            return

        # Last all objects
        for obj in reversed(comp.ode_objects):
            if isinstance(obj, self._types):
                yield obj

    def _iter_objects(self, comp):

        # First return component
        if self._return_comp:
            yield comp

        # Secondly all objects
        if not self._only_return_comp:

            for obj in comp.ode_objects:
                if isinstance(obj, self._types):
                    yield obj

        # Thrirdly all children components
        for sub_comp in list(comp.children.values()):
            for sub_tree in self._iter_objects(sub_comp):
                yield sub_tree

    def __next__(self):
        return next(self._object_iterator)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()


def ode_objects(comp, *types):
    """
    Return a list of ode objects

    Arguments
    ---------
    comp : gotran.ODEComponent
        The root ODEComponent of the list
    types : gotran.ODEObject types (optional)
        Only include objects of type given in types
    """
    return [obj for obj in iter_objects(comp, False, False, False, *types)]


def ode_components(comp, include_self=True):
    """
    Return a list of ode components

    Arguments
    ---------
    comp : gotran.ODEComponent
        The root ODEComponent of the list
    return_self : bool (optional)
        The list will include the passed component if True
    """
    comps = [obj for obj in iter_objects(comp, True, True)]

    if not include_self:
        comps.remove(comp)

    return comps


class ODEObjectList(list):
    """
    Specialized container for ODEObjects. It is a list but adds dict
    access through the name attribute of an ODEObjects
    """

    def __init__(self):
        """
        Initialize ODEObjectList. Only empty such.
        """
        super(ODEObjectList, self).__init__()
        self._objects = {}

    def keys(self):
        return list(self._objects.keys())

    def append(self, item):
        check_arg(item, ODEObject, 0, ODEObjectList.append)
        super(ODEObjectList, self).append(item)
        self._objects[item.name] = item

    def insert(self, index, item):
        check_arg(item, ODEObject, 1, ODEObjectList.insert)
        super(ODEObjectList, self).insert(index, item)
        self._objects[item.name] = item

    def extend(self, iterable):
        check_arg(iterable, list, 0, ODEObjectList.extend, ODEObject)
        super(ODEObjectList, self).extend(iterable)
        for item in iterable:
            self._objects[item.name] = item

    def get(self, name):
        if isinstance(name, str):
            return self._objects.get(name)
        elif isinstance(name, sp.Symbol):
            return self._objects.get(name.name)
        return None

    def __contains__(self, item):
        if isinstance(item, str):
            return any(item == obj.name for obj in self)
        elif isinstance(item, sp.Symbol):
            return any(item.name == obj.name for obj in self)
        elif isinstance(item, ODEObject):
            return super(ODEObjectList, self).__contains__(item)
        return False

    def count(self, item):
        if isinstance(item, str):
            return sum(item == obj.name for obj in self)
        elif isinstance(item, sp.Symbol):
            return sum(item.name == obj.name for obj in self)
        elif isinstance(item, ODEObject):
            return super(ODEObjectList, self).count(item)
        return 0

    def index(self, item):
        if isinstance(item, str):
            for ind, obj in enumerate(self):
                if item == obj.name:
                    return ind
        elif isinstance(item, sp.Symbol):
            for ind, obj in enumerate(self):
                if item.name == obj.name:
                    return ind
        elif isinstance(item, ODEObject):
            for ind, obj in enumerate(self):
                if item == obj:
                    return ind
        raise ValueError(f"Item '{str(item)}' not part of this ODEObjectList.")

    def sort(self):
        error("Cannot sort ODEObjectList.")

    def pop(self, index):

        check_arg(index, int)
        if index >= len(self):
            raise IndexError("pop index out of range")
        obj = super(ODEObjectList, self).pop(index)
        self._objects.pop(obj.name)

    def remove(self, item):
        try:
            index = self.index(item)
        except ValueError:
            raise ValueError("ODEObjectList.remove(x): x not in list")

        self.pop(index)

    def reverse(self, item):
        error("Cannot alter ODEObjectList, other than adding ODEObjects.")


class RateDict(OrderedDict):
    """
    A storage class for Markov model rates
    """

    def __init__(self, comp):
        from .odecomponent import ODEComponent

        check_arg(comp, ODEComponent)
        self._comp = weakref.ref(comp)
        super(RateDict, self).__init__()

    def __setitem__(self, states, expr):
        """
        Register rate(s) between states

        Arguments
        ---------
        states : tuple of two states, list of States, tuple of two lists of States
            If tuple of two states is given a single rate is expected.
            If one list is passed the rate expression should be a square matrix
            and the states list determines the order of the row and column of
            the matrix. If two lists are passed the first determines the states
            in the row and the second the states in the column of the Matrix
        expr : sympy.Basic, scalar, sympy.MatrixBase
            A sympy.Basic and scalars is expected for a single rate between
            two states.
            A sympy.Matrix is expected if several rate expressions are given,
            see explaination in for states argument for how the columns and
            rows are interpreted.
        """

        if isinstance(expr, sp.Matrix):
            self._comp()._add_rates(states, expr)
        else:
            if not isinstance(states, tuple) or len(states) != 2:
                error(
                    "Expected a tuple of size 2 with states when "
                    "registering a single rate.",
                )

            # NOTE: the actuall item is set by the component while calling this
            # function, using _register_single_rate. See below.
            self._comp()._add_single_rate(states[0], states[1], expr)

    def _register_single_rate(self, to_state, from_state, expr_sym):
        """
        Actually setting an item
        """
        OrderedDict.__setitem__(self, (to_state, from_state), expr_sym)
