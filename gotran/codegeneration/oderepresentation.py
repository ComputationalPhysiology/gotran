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
from collections import deque, OrderedDict
import hashlib

# Model parametrs imports
from modelparameters.parameterdict import *
from modelparameters.sympytools import sp, iter_symbol_params_from_expr

# Local gotran imports
from gotran.model.ode import ODE
from gotran.model.odecomponents import ODEComponent, Comment
from gotran.common import check_arg, check_kwarg, info

import sys

def _default_params(exclude=None):
    exclude = exclude or []
    check_arg(exclude, list, itemtypes=str)

    # Add not implemented parameters to excludes
    exclude += ["max_terms", "parameter_contraction", "use_variables"]

    # Build a dict with allowed parameters
    params = {}
    
    if "use_state_names" not in exclude:
        # Use state names in code (compared to array with indices)
        params["use_state_names"] = Param(True, \
                                          description="Use state names in code "\
                                          "(compared to array with indices)")

    if "use_parameter_names" not in exclude:
        # Use parameter names in code (compared to array with indices)
        params["use_parameter_names"] = Param(\
            True, description="Use parameter names "\
            "in code (compared to array with indices)")

    if "keep_intermediates" not in exclude:
        # Keep all intermediates
        params["keep_intermediates"] = Param(\
            True, description="Keep intermediates in code")

    if "use_variables" not in exclude:
        # If True, code for altering variables are created
        # FIXME: Not used
        params["use_variables"] = Param(\
            False, description="If True, code for altering variables are created")

    if "parameter_contraction" not in exclude:
        # Find sub expressions of only parameters and create a dummy parameter
        # FIXME: Not used
        params["parameter_contraction"] = Param(\
            False, description="Find sub expressions of only parameters "\
            "and create a dummy parameter")

    if "parameter_numerals" not in exclude:
        # Exchange all parameters with their initial numerical values
        params["parameter_numerals"] = Param(\
            False, description="Exchange all parameters with their initial"\
            " numerical values")

    if "max_terms" not in exclude:
        # Split terms with more than max_terms into several evaluations
        # FIXME: Not used
        params["max_terms"] = ScalarParam(\
            5, ge=2, description="Split terms with more than max_terms "\
            "into several evaluations")

    if "use_cse" not in exclude:
        # Use sympy common sub expression simplifications,
        # only when keep_intermediates is false
        params["use_cse"] = Param(\
            False, description="Use sympy common sub expression "\
            "simplifications, only when keep_intermediates is false")

    # Return the ParameterDict
    return ParameterDict(**params)

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

        self._used_in_monitoring = None
        self._cse_subs = None

    def signature(self):
        # Create a unique signature
        h = hashlib.sha1()
        h.update(self.ode.signature() + repr(self.optimization))
        return h.hexdigest()
    
    def _compute_dy_cse(self):
        if self._cse_subs is not None:
            return

        ode = self.ode

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
                if self._cse_subs[i][0] in self._cse_subs[j][1].atoms():
                    cse_counts[i].append(j)
            
            for j in range(len(self._cse_derivative_expr)):
                if self._cse_subs[i][0] in self._cse_derivative_expr[j].atoms():
                    cse_counts[i].append(j+len(self._cse_subs))

        # Store usage count
        # FIXME: Use this for more sorting!
        self._cse_counts = cse_counts

        info(" done")
        
    def _compute_monitor_cse(self):
        if self._used_in_monitoring is not None:
            return

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

    def _compute_jacobian(self):
       pass 
            
    def update_index(self, index):
        """
        Set index notation, specific for language syntax
        """
        self.index = index

    @property
    def name(self):
        return self._name

    @property
    def class_name(self):
        name = self.name
        return name if name[0].isupper() else name[0].upper() + \
               (name[1:] if len(name) > 1 else "")    

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

            subs = []
            
            # Deal with parameter subs first
            if self.optimization.parameter_numerals:
                subs.extend((param.sym, param.init) \
                            for param in self.ode.parameters)
            elif not self.optimization.use_parameter_names:
                subs.extend((param.sym, sp.Symbol("parameters"+self.index(\
                    ind))) for ind, param in enumerate(\
                                self.ode.parameters))
            elif self._parameter_prefix:
                subs.extend((param.sym, sp.Symbol("{0}{1}".format(\
                    self._parameter_prefix, param.name))) \
                            for param in self.ode.parameters)

            # Deal with state subs
            if not self.optimization.use_state_names:
                subs.extend((state.sym, sp.Symbol("states"+self.index(ind)))\
                            for ind, state in enumerate(\
                                self.ode.states))

            elif self._state_prefix:
                subs.extend((param.sym, sp.Symbol("{0}{1}".format(\
                    self._state_prefix, param.name))) \
                            for param in self.ode.states)

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
            self._compute_dy_cse()
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
            for intermediate in self.ode.intermediates:
                if isinstance(intermediate, (Comment, ODEComponent)):
                    yield intermediate.name, "COMMENT"
                    continue
                
                yield self.subs(intermediate.expr), intermediate.name
                    
        elif self.optimization.use_cse:
            self._compute_dy_cse()
            yield "Common Sub Expressions", "COMMENT"
            for (name, expr), cse_count in zip(self._cse_subs, self._cse_counts):
                if cse_count:
                    yield expr, name

    def iter_monitored_body(self):
        """
        Iterate over the body defining the monitored expressions 
        """
        
        self._compute_monitor_cse()
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
        
        self._compute_monitor_cse()
        if self.ode.num_monitored_intermediates == 0:
            return

        yield "COMMENT", "Monitored intermediates"

        for ((name, expr), cse_expr) in zip(\
            self.ode.iter_monitored_intermediates(), \
            self._cse_monitored_expr):
            yield name, cse_expr
