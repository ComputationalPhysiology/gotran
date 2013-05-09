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
import re
import sys


# Model parametrs imports
from modelparameters.parameterdict import *
from modelparameters.sympytools import sp, iter_symbol_params_from_expr

# Local gotran imports
from gotran.model.ode import ODE
from gotran.model.odecomponents import ODEComponent, Comment
from gotran.common import check_arg, check_kwarg, info

_jacobian_pattern = re.compile("_([0-9]+)")

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

    if "generate_jacobian" not in exclude:
        # Generate code for the computation of the jacobian
        params["generate_jacobian"] = Param(\
            True, description="Generate code for the computation of the jacobian")
    
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
        self._jacobian_expr = None
        self._cse_subs_single_dy = None
        self._linear_terms = None

    @property
    def used_in_monitoring(self):
        if self._used_in_monitoring is None:
            self._compute_monitor_cse()
        return self._used_in_monitoring

    @property
    def used_in_linear_dy(self):
        if self._linear_terms is None:
            self._compute_linearized_dy()
        
        return self._used_in_linear_dy

    @property
    def used_in_single_dy(self):
        if self._linear_terms is None:
            self._compute_linearized_dy()
        return self._used_in_single_dy

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

        ode = self.ode

        for name, obj in ode.monitored_intermediates.items():
            for sym in iter_symbol_params_from_expr(obj.expanded_expr):

                if ode.has_state(sym):
                    self._used_in_monitoring["states"].add(sym.name)
                elif ode.has_parameter(sym):
                    self._used_in_monitoring["parameters"].add(sym.name)
                else:
                    # Skip if ModelSymbols is not state or parameter
                    pass
        
        self._cse_monitored_subs, self._cse_monitored_expr = \
                    sp.cse([self.subs(obj.expanded_expr) \
                        for obj in ode.monitored_intermediates.values()], \
                           symbols=sp.numbered_symbols("cse_monitored_"), \
                           optimizations=[])

        cse_counts = [[] for i in range(len(self._cse_monitored_subs))]
        for i in range(len(self._cse_monitored_subs)):
            for j in range(i+1, len(self._cse_monitored_subs)):
                if self._cse_monitored_subs[i][0] in \
                       self._cse_monitored_subs[j][1].atoms():
                    cse_counts[i].append(j)
                
            for j in range(len(self._cse_monitored_expr)):
                if self._cse_monitored_subs[i][0] in self._cse_monitored_expr[j].atoms():
                    cse_counts[i].append(j+len(self._cse_monitored_subs))

        # Store usage count
        # FIXME: Use this for more sorting!
        self._cse_monitored_counts = cse_counts

        self._used_in_monitoring["parameters"] = \
                            list(self._used_in_monitoring["parameters"])

        self._used_in_monitoring["states"] = \
                            list(self._used_in_monitoring["states"])

    def _compute_jacobian_cse(self):

        if self._jacobian_expr is not None:
            return 

        ode = self.ode

        sym_map = OrderedDict()
        jacobi_expr = OrderedDict()
        for i, (ders, expr) in enumerate(ode.get_derivative_expr(True)):
            for j, state in enumerate(ode.states):
                F_ij = expr.diff(state.sym)

                # Only collect non zero contributions
                if F_ij:
                    jacobi_sym = sp.Symbol("j_{0}_{1}".format(i,j))
                    sym_map[i,j] = jacobi_sym
                    jacobi_expr[jacobi_sym] = F_ij
                    #print "[%d,%d] (%d) # [%s, %s]  \n%s" \
                    # % (i,j, len(F_ij.args), states[i], states[j], F_ij)
            
        # Create the Jacobian
        self._jacobian_mat = sp.SparseMatrix(ode.num_states, ode.num_states, \
                                           lambda i, j: sym_map.get((i, j), 0))
        self._jacobian_expr = jacobi_expr

        if not self.optimization.use_cse:
            return

        info("Calculating jacobi common sub expressions for {0} entries. "\
             "May take some time...".format(len(self._jacobian_expr)))
        sys.stdout.flush()
        # If we use cse we extract the sub expressions here and cache
        # information
        self._cse_jacobian_subs, self._cse_jacobian_expr = \
                sp.cse([self.subs(expr) \
                        for expr in self._jacobian_expr.values()], \
                       symbols=sp.numbered_symbols("cse_jacobian_"), \
                       optimizations=[])
        
        cse_jacobian_counts = [[] for i in range(len(self._cse_jacobian_subs))]
        for i in range(len(self._cse_jacobian_subs)):
            for j in range(i+1, len(self._cse_jacobian_subs)):
                if self._cse_jacobian_subs[i][0] in self._cse_jacobian_subs[j][1].atoms():
                    cse_jacobian_counts[i].append(j)
            
            for j in range(len(self._cse_jacobian_expr)):
                if self._cse_jacobian_subs[i][0] in self._cse_jacobian_expr[j].atoms():
                    cse_jacobian_counts[i].append(j+len(self._cse_jacobian_subs))

        # Store usage count
        # FIXME: Use this for more sorting!
        self._cse_jacobian_counts = cse_jacobian_counts

        info(" done")

    def _compute_linearized_dy_cse(self):
        self._compute_linearized_dy()
        
        if self._cse_subs_single_dy is not None:
            return

        info("Calculating common sub expressions for single components of ODE. May take some time...")
        sys.stdout.flush()
        ode = self.ode

        self._cse_subs_single_dy = []
        self._cse_derivative_expr_single_dy = []
        
        # Iterate over the derivative terms and collect information
        for i, (ders, expr) in enumerate(ode.get_derivative_expr(True)):
            cse_subs, cse_derivative_expr = \
                      sp.cse(self.subs(expr), \
                             symbols=sp.numbered_symbols("cse_der_{0}_".format(i)),\
                             optimizations=[])
            self._cse_subs_single_dy.append(cse_subs)
            self._cse_derivative_expr_single_dy.append(cse_derivative_expr)
            
        info(" done")
        info("Calculating common sub expressions for linearized ODE. May take some time...")
        self._cse_linearized_subs, self._cse_linearized_derivative_expr = \
                                   sp.cse([self.subs(expr) \
                                           for expr in self._linearized_exprs.values()], \
                                          symbols=sp.numbered_symbols("cse_linear_"), \
                                          optimizations=[])

        info(" done")
        
    def _compute_linearized_dy(self):
        if self._linear_terms is not None:
            return

        ode = self.ode

        used_in_single_dy = []
        used_in_linear_dy = dict(parameters=set(), states=set())
        
        linearized_exprs = OrderedDict()

        # Iterate over the derivative terms and collect information
        for i, (ders, expr) in enumerate(ode.get_derivative_expr(True)):
            used_states = set()
            used_parameters = set()
            for sym in iter_symbol_params_from_expr(expr):
                if ode.has_state(sym):
                    used_states.add(sym.name)
                elif ode.has_parameter(sym):
                    used_parameters.add(sym.name)

            used_in_single_dy.append(dict(parameters=list(used_parameters),
                                          states=list(used_states)))
            
            assert(len(ders)==1)
            
            # Grab state for the derivative
            state_sym = ders[0].sym
            expr_diff = expr.diff(state_sym)
            
            # Check for linear term
            if expr_diff and state_sym not in expr_diff.atoms():

                linearized_exprs[i] = expr_diff
            
                for sym in iter_symbol_params_from_expr(expr_diff):
                    if ode.has_state(sym):
                        used_in_linear_dy["states"].add(sym.name)
                    elif ode.has_parameter(sym):
                        used_in_linear_dy["parameters"].add(sym.name)

        # Store data    
        self.linear_terms = [i in linearized_exprs for i in range(ode.num_states)]
        self._used_in_single_dy = used_in_single_dy
        self._linearized_exprs = linearized_exprs
        self._used_in_linear_dy = dict(states = list(used_in_linear_dy["states"]),
                                      parameters = list(used_in_linear_dy["parameters"]))
            
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

    def iter_jacobian_body(self):
        """
        Iterate over the body defining the jacobi expressions 
        """
        
        if not self.optimization.use_cse:
            return 
            
        self._compute_jacobian_cse()

        # Yield the CSE
        yield "Common Sub Expressions for jacobi intermediates", "COMMENT"
        for (name, expr), cse_count in zip(self._cse_jacobian_subs, \
                                           self._cse_jacobian_counts):
            if cse_count:
                yield expr, name

    def iter_jacobian_expr(self):
        """
        Iterate over the jacobi expressions 
        """
        
        self._compute_jacobian_cse()

        if self.optimization.use_cse:
            
            for ((name, expr), cse_expr) in zip(\
                self._jacobian_expr.items(), \
                self._cse_jacobian_expr):
                yield map(int, re.findall(_jacobian_pattern, str(name))), cse_expr
        else:

            for name, expr in self._jacobian_expr.items():
                yield map(int, re.findall(_jacobian_pattern, str(name))), self.subs(expr)
            
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

        for (name, cse_expr) in zip(\
            self.ode.monitored_intermediates, \
            self._cse_monitored_expr):
            yield name, cse_expr

    def iter_linerized_body(self):
        if not self.optimization.use_cse:
            return 
            
        self._compute_linearized_dy_cse()

        # Yield the CSE
        yield "Common Sub Expressions for linearized evaluations", "COMMENT"
        for name, expr in self._cse_linearized_subs:
            yield expr, name
        
    def iter_linerized_expr(self):
        
        if self.optimization.use_cse:
            self._compute_linearized_dy_cse()
            for idx, cse_expr in zip(self._linearized_exprs, \
                                     self._cse_linearized_derivative_expr):
                yield idx, cse_expr

        else:
            self._compute_linearized_dy()
            for idx, expr in self._linearized_exprs.items():
                yield idx, expr
            
    def iter_componentwise_dy(self):
        
        if self.optimization.use_cse:
            self._compute_linearized_dy_cse()
            for subs, expr in zip(self._cse_subs_single_dy, \
                                  self._cse_derivative_expr_single_dy):
                yield subs, expr[0]

        else:
            self._compute_linearized_dy()
            for ders, expr in self.ode.get_derivative_expr(True):
                yield [], expr
