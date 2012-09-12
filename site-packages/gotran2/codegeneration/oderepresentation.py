from collections import deque

from modelparameters.parameterdict import *
from modelparameters.sympytools import sp

from gotran2.model.ode import ODE
from gotran2.common import check_arg

def _default_params():
    return ParameterDict(

        # Use state, parameters, and variable names in code (compared to
        # array with indices)
        use_names = True,

        # Keep all intermediates
        keep_intermediates = True, 

        # If True, logic for field states are created
        field_states = False,

        # If True, logic for field paramters are created
        field_parameters = False,

        # If True , code for altering variables are created
        use_variables = False,

        # Find sub expressions of only parameters and create a dummy parameter
        parameter_contraction = False,

        # Exchange all parameters with their initial numerical values
        parameter_numerals = False,

        # Split terms with more than max_terms into several evaluations
        max_terms = ScalarParam(5, ge=2),

        # Use sympy common sub expression simplifications
        use_cse = False,
        )

class ODERepresentation(object):
    """
    Intermediate ODE representation where various optimizations
    can be performed.
    """
    def __init__(self, ode, **optimization):
        check_arg(ode, ODE, 0)
        
        self.ode = ode
        self.optimization = _default_params()
        self.optimization.update(optimization)
        self._symbol_subs = None
        self.index = lambda i : "[{0}]".format(i)

        # Check for using CSE
        if not self.optimization.keep_intermediates and \
               self.optimization.use_cse:

            # If we use cse we extract the sub expressions here and cache information
            self._cse_subs, self._cse_derivative_expr = \
                    sp.cse([self.subs(expr) \
                            for der, expr in ode.get_derivative_expr(True)], \
                           symbols=sp.numbered_symbols("cse_"), optimizations=[])
            cse_counts = [[] for i in range(len(self._cse_subs))]
            for i in range(len(self._cse_subs)):
                for j in range(i+1, len(self._cse_subs)):
                    if self._cse_subs[i][0] in self._cse_subs[j][1]:
                        cse_counts[i].append(j)
                    
                for j in range(len(self._cse_derivative_expr)):
                    if self._cse_subs[i][0] in self._cse_derivative_expr[j]:
                        cse_counts[i].append(j+len(self._cse_subs))

            # Store usage count
            # FIXME: Use this for more sorting!
            self._cse_counts = cse_counts
        
    def update_index(self, index):
        """
        Set index notation, specific for language syntax
        """
        self.index = index

    @property
    def name(self):
        return self.ode.name

    def subs(self, expr):
        if isinstance(expr, sp.Basic):
            return expr.subs(self.symbol_subs)
        return expr
    
    @property
    def symbol_subs(self):
        """
        return a subs dict for parameters
        """
        if self._symbol_subs is None:

            subs = {}
            # Deal with parameter subs first
            if self.optimization.parameter_numerals:
                subs.update((param.sym, param.init) \
                            for param in self.ode.iter_parameters())
            elif not self.optimization.use_names:
                subs.update((param.sym, sp.Symbol("parameters"+self.index(ind)))\
                            for ind, param in enumerate(self.ode.iter_parameters()))

            # Deal with state subs
            if not self.optimization.use_names:
                subs.update((state.sym, sp.Symbol("states"+self.index(ind)))\
                            for ind, state in enumerate(self.ode.iter_states()))

            self._symbol_subs = subs
                
        return self._symbol_subs

    def iter_derivative_expr(self):
        """
        Return a list of derivatives and its expressions
        """

        # Keep intermediates is the lowest form for optimization deal with first
        if self.optimization.keep_intermediates:

            return ((derivatives, self.subs(expr)) \
                    for derivatives, expr in self.ode.get_derivative_expr())

        # No intermediates and no CSE
        if not self.optimization.use_cse:
            return ((derivatives, self.subs(expr)) \
                    for derivatives, expr in self.ode.get_derivative_expr(True))

        # Use CSE
        else:
            return ((derivatives, cse_expr) \
                    for ((derivatives, expr), cse_expr) in zip(\
                        self.ode.get_derivative_expr(), self._cse_derivative_expr))
            
        

    def iter_dy_body(self):
        """
        Return an interator over dy_body lines

        If using intermediates it will define these,
        if using cse extraction these will be returned
        """
        
        if self.optimization.keep_intermediates:

            # Iterate over the intermediates
            intermediates_duplicates = dict((intermediate, deque(duplicates)) \
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
            for name, expr in self._cse_subs:
                yield expr, name
            

    


