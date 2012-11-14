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

# System imports
from collections import deque

# Model parametrs imports
from modelparameters.parameterdict import *
from modelparameters.sympytools import sp, iter_symbol_params_from_expr

# Local gotran imports
from gotran.model.ode import ODE
from gotran.common import check_arg, check_kwarg, info

import sys

def _default_params():
    return ParameterDict(

        # Use state names in code (compared to array with indices)
        use_state_names = 1,

        # Use parameter names in code (compared to array with indices)
        use_parameter_names = 1,

        # Keep all intermediates
        keep_intermediates = 1, 

        # If True, code for altering variables are created
        # FIXME: Not used
        use_variables = 0,

        # Find sub expressions of only parameters and create a dummy parameter
        # FIXME: Not used
        parameter_contraction = 0,

        # Exchange all parameters with their initial numerical values
        parameter_numerals = 0,

        # Split terms with more than max_terms into several evaluations
        # FIXME: Not used
        max_terms = ScalarParam(5, ge=2),

        # Use sympy common sub expression simplifications,
        # only when keep_intermediates is false
        use_cse = 0,
        )

class ODERepresentation(object):
    """
    Intermediate ODE representation where various optimizations
    can be performed.
    """
    def __init__(self, ode, name="", **optimization):
        """
        Create an ODERepresentation

        Arguments:
        ----------
        ode : ODE
            The ode to be represented
        name : str (optional)
            An argument which determines the name of the ode representation,
            making it possible to create different representations identified
            with different names.
        """
        check_arg(ode, ODE, 0)
        check_kwarg(name, "name", str)

        self.ode = ode

        # Store the name
        self._name = name if name else ode.name
        
        self.optimization = _default_params()

        self.optimization.update(optimization)
        self._symbol_subs = None
        self.index = lambda i : "[{0}]".format(i)

        # Init prefix info
        self._state_prefix = ""
        self._parameter_prefix = ""

        # Check for using CSE
        if not self.optimization.keep_intermediates and \
               self.optimization.use_cse:

            info("Calculating common sub expressions. May take some time...")
            sys.stdout.flush()
            # If we use cse we extract the sub expressions here and cache
            # information
            self._cse_subs, self._cse_derivative_expr = \
                    sp.cse([self.subs(expr) \
                            for der, expr in ode.get_derivative_expr(True)], \
                           symbols=sp.numbered_symbols("cse_"), \
                           optimizations=[])
            
            cse_counts = [[] for i in range(len(self._cse_subs))]
            for i in range(len(self._cse_subs)):
                for j in range(i+1, len(self._cse_subs)):
                    if self._cse_subs[i][0] in self._cse_subs[j][1]:
                        cse_counts[i].append(j)
                
                for j in range(len(self._cse_derivative_expr)):
                    if self._cse_subs[i][0] in self._cse_derivative_expr[j].atoms():
                        cse_counts[i].append(j+len(self._cse_subs))

            # Store usage count
            # FIXME: Use this for more sorting!
            self._cse_counts = cse_counts

            info(" done")

        # If the ode contains any monitored intermediates generate CSE versions 
        if ode.num_monitored_intermediates > 0:

            # Generate info about used states and parameters
            self._used_in_monitoring = dict(
                states = set(), parameters = set()
                )

            for name, expr in ode.iter_monitored_intermediates():
                for sym in iter_symbol_params_from_expr(expr):

                    if ode.has_state(sym):
                        self._used_in_monitoring["states"].add(sym.name)
                    elif ode.has_parameter(sym):
                        self._used_in_monitoring["parameters"].add(sym.name)
                    else:
                        # Skip if ModelSymbols is not state or parameter
                        pass
            
            self._cse_monitored_subs, self._cse_monitored_expr = \
                        sp.cse([self.subs(expr) \
                            for name, expr in ode.iter_monitored_intermediates()], \
                               symbols=sp.numbered_symbols("cse_"), \
                               optimizations=[])
            
            cse_counts = [[] for i in range(len(self._cse_monitored_subs))]
            for i in range(len(self._cse_monitored_subs)):
                for j in range(i+1, len(self._cse_monitored_subs)):
                    if self._cse_monitored_subs[i][0] in \
                           self._cse_monitored_subs[j][1]:
                        cse_counts[i].append(j)
                    
                for j in range(len(self._cse_monitored_expr)):
                    if self._cse_monitored_subs[i][0] in self._cse_monitored_expr[j]:
                        cse_counts[i].append(j+len(self._cse_monitored_subs))

            # Store usage count
            # FIXME: Use this for more sorting!
            self._cse_monitored_counts = cse_counts

            self._used_in_monitoring["parameters"] = \
                                list(self._used_in_monitoring["parameters"])

            self._used_in_monitoring["states"] = \
                                list(self._used_in_monitoring["states"])
            
    def update_index(self, index):
        """
        Set index notation, specific for language syntax
        """
        self.index = index

    @property
    def name(self):
        return self._name

    def set_state_prefix(self, prefix):
        """
        Register a prefix to a state name. Used if 
        """
        check_arg(prefix, str)
        self._state_prefix = prefix

        # Reset symbol subs
        self._symbol_subs = None

    def set_parameter_prefix(self, prefix):
        """
        Register a prefix to a parameter name. Used if 
        """
        check_arg(prefix, str)
        self._parameter_prefix = prefix

        # Reset symbol subs
        self._symbol_subs = None

    def subs(self, expr):
        """
        Call subs on the passed expr using symbol_subs if the expr is
        a SymPy Basic
        """
        if isinstance(expr, sp.Basic):
            return expr.subs(self.symbol_subs)
        return expr
    
    @property
    def symbol_subs(self):
        """
        Return a subs dict for all ODE Objects (states, parameters)
        """
        if self._symbol_subs is None:

            subs = {}
            
            # Deal with parameter subs first
            if self.optimization.parameter_numerals:
                subs.update((param.sym, param.init) \
                            for param in self.ode.iter_parameters())
            elif not self.optimization.use_parameter_names:
                subs.update((param.sym, sp.Symbol("parameters"+self.index(\
                    ind))) for ind, param in enumerate(\
                                self.ode.iter_parameters()))
            elif self._parameter_prefix:
                subs.update((param.sym, sp.Symbol("{0}{1}".format(\
                    self._parameter_prefix, param.name))) \
                            for param in self.ode.iter_parameters())

            # Deal with state subs
            if not self.optimization.use_state_names:
                subs.update((state.sym, sp.Symbol("states"+self.index(ind)))\
                            for ind, state in enumerate(\
                                self.ode.iter_states()))

            elif self._state_prefix:
                subs.update((param.sym, sp.Symbol("{0}{1}".format(\
                    self._state_prefix, param.name))) \
                            for param in self.ode.iter_states())

            self._symbol_subs = subs
                
        return self._symbol_subs

    def iter_derivative_expr(self):
        """
        Return a list of derivatives and its expressions
        """

        # Keep intermediates is the lowest form for optimization deal with
        # first
        if self.optimization.keep_intermediates:

            return ((derivatives, self.subs(expr)) \
                    for derivatives, expr in self.ode.get_derivative_expr())

        # No intermediates and no CSE
        if not self.optimization.use_cse:
            return ((derivatives, self.subs(expr)) \
                    for derivatives, expr in self.ode.get_derivative_expr(\
                        True))

        # Use CSE
        else:
            return ((derivatives, cse_expr) \
                    for ((derivatives, expr), cse_expr) in zip(\
                        self.ode.get_derivative_expr(), \
                        self._cse_derivative_expr))

    def iter_dy_body(self):
        """
        Return an interator over dy_body lines

        If using intermediates it will define these,
        if using cse extraction these will be returned
        """
        
        if self.optimization.keep_intermediates:

            # Iterate over the intermediates
            intermediates_duplicates = dict((intermediate, deque(duplicates))\
                                    for intermediate, duplicates in \
                                    self.ode._intermediates_duplicates.items())
            for intermediate, expr in self.ode._intermediates.items():
                if "_comment_" in intermediate:
                    yield expr, "COMMENT"
                    continue
                
                if "_duplicate_" in intermediate:
                    # expr is here a str of the name which can be used as key
                    # in duplicate dictionary
                    intermediate = expr
                    expr = intermediates_duplicates[intermediate].popleft()

                yield self.subs(expr), intermediate
                    
        elif self.optimization.use_cse:
            yield "Common Sub Expressions", "COMMENT"
            for (name, expr), cse_count in zip(self._cse_subs, self._cse_counts):
                if cse_count:
                    yield expr, name

    def iter_monitored_body(self):
        """
        Iterate over the body defining the monitored expressions 
        """
        
        if self.ode.num_monitored_intermediates == 0:
            return

        # Yield the CSE
        yield "Common Sub Expressions for monitored intermediates", "COMMENT"
        for (name, expr), cse_count in zip(self._cse_monitored_subs, \
                                           self._cse_monitored_counts):
            if cse_count:
                yield expr, name

    def iter_monitored_expr(self):
        """
        Iterate over the monitored expressions 
        """
        
        if self.ode.num_monitored_intermediates == 0:
            return

        yield "COMMENT", "Monitored intermediates"

        for ((name, expr), cse_expr) in zip(\
            self.ode.iter_monitored_intermediates(), \
            self._cse_monitored_expr):
            yield name, cse_expr
