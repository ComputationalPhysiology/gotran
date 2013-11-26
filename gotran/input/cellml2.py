# Copyright (C) 2011-2012 Johan Hake
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

import sys, os, re
import urllib
from xml.etree import ElementTree

from collections import OrderedDict, deque
from gotran.common import warning, error, check_arg

from modelparameters.codegeneration import _all_keywords
from modelparameters.parameterdict import *

__all__ = ["cellml2ode", "CellMLParser"]

ui = "UNINITIALIZED"

class Equation(object):
    """
    Class for holding information about an Equation
    """
    def __init__(self, name, expr, used_variables):
        self.name = name
        self.expr = expr
        self.used_variables = used_variables
        self.dependent_equations = []
        self.component = None

    def check_dependencies(self, equation):
        """
        Check Equation dependencies
        """
        assert(isinstance(equation, Equation))

        if equation.name in self.used_variables:
            self.dependent_equations.append(equation)
            
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return "Equation({0} = {1})".format(self.name, "".join(self.expr))

    def __eq__(self, other):
        if not isinstance(other, Equation):
            return False
        return other.name == self.name and other.component == self.component

    def __hash__(self):
        return hash(self.name+self.component.name)

class Component(object):
    """
    Class for holding information about a CellML Component
    """
    def __init__(self, name, variables, equations, state_variables=None):
        self.name = name

        self.state_variables = OrderedDict((state, variables.pop(state, None))\
                                           for state in state_variables)

        self.parameters = OrderedDict((name, value) for name, value in \
                                      variables.items() if value is not None)

        self.parent = None
        
        self.derivatives = state_variables

        self.components_dependencies = OrderedDict() 
        self.dependent_components = OrderedDict() 

        # Attributes which will be populated later
        # FIXME: Should we populate the parameters based on the variables
        # FIXME: with initial values which are not state variables

        # Check internal dependencies
        for eq0 in equations:
            
            # Store component
            eq0.component = self
            for eq1 in equations:
                if eq0 == eq1:
                    continue
                eq0.check_dependencies(eq1)

        sorted_equations = []
        while equations:
            equation = equations.pop(0)
            if any(dep in equations for dep in equation.dependent_equations):
                equations.append(equation)
            else:
                sorted_equations.append(equation)

        # Store the sorted equations
        self.equations = sorted_equations

        # Get used variables
        self.used_variables = set()
        for equation in self.equations:
            self.used_variables.update(equation.used_variables)

        # Remove dependencies on names defined by component
        self.used_variables.difference_update(\
            equation.name for equation in self.equations)

        self.used_variables.difference_update(\
            name for name in self.parameters)

        self.used_variables.difference_update(\
            name for name in self.state_variables)

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name + "<{0}>".format(len(self.state_variables))

    def __repr__(self):
        return "Component<{0}, {1}>".format(self.name, \
                                            len(self.state_variables))

    def __eq__(self, other):
        if not isinstance(other, Component):
            return False
        
        return other.name == self.name

    def check_dependencies(self, component):
        """
        Check components dependencies
        """
        assert(isinstance(component, Component))
        
        if any(equation.name in self.used_variables \
               for equation in component.equations):
            dep_equations = [equation for equation in component.equations \
                             if equation.name in self.used_variables]

            # Register mutual dependencies
            self.components_dependencies[component] = dep_equations
            component.dependent_components[self] = dep_equations
            
            # Add logics for dependencies of all Equations.
            for other_equation in component.equations:
                for equation in self.equations:
                    if other_equation.name in equation.used_variables:
                        equation.dependent_equations.append(other_equation)

    def change_parameter_name(self, oldname, newname):
        """
        Change the name of a parameter
        Assume the name is only used locally within this component
        """
        assert(oldname in self.parameters)
        
        warning("Locally change parameter name: '{0}' to '{1}' in "\
                "component '{2}'.".format(oldname, newname, self.name))
        
        # Update parameters
        self.parameters = OrderedDict((newname if name == oldname else \
                                       name, value) for name, value in \
                                      self.parameters.items())

        # Update equations
        for eqn in self.equations:
            while oldname in eqn.expr:
                eqn.expr[eqn.expr.index(oldname)] = newname

    def change_state_name(self, oldname, newname):
        """
        Change the name of a state
        Assume the name is only used locally within this component
        """
        assert(oldname in self.state_variables)
        
        warning("Locally change state name: '{0}' to '{1}' in component "\
                "'{2}'.".format(oldname, newname, self.name))
        
        # Update parameters
        self.state_variables = OrderedDict((newname if name == oldname \
                            else name, value) for name, value in \
                            self.state_variables.items())

        oldder = self.derivatives[oldname]
        newder = oldder.replace(oldname, newname)
        self.derivatives = OrderedDict((newname if name == oldname else name, \
                                        newder if value == oldder else value) \
                                       for name, value in \
                                       self.derivatives.items())
        # Update equations
        for eqn in self.equations:
            while oldname in eqn.expr:
                eqn.expr[eqn.expr.index(oldname)] = newname
            while oldder in eqn.expr:
                eqn.expr[eqn.expr.index(oldder)] = newder
            if oldder == eqn.name:
                eqn.name = newder

class MathMLBaseParser(object):
    def __init__(self):
        self._state_variable = None
        self._derivative = None
        self.variables_names = set()
    
        self._precedence = {
            "piecewise" : 0, 
            "power" : 0,
            "divide": 1,
            "times" : 2,
            "minus" : 4,
            "plus"  : 5,
            "lt"    : 6,
            "gt"    : 6,
            "leq"   : 6,
            "geq"   : 6,
            "eq"    : 10,
            "exp"   : 10,
            "ln"    : 10,
            "abs"   : 10,
            "floor" : 10,
            "log"   : 10,
            "root"  : 10,
            "tan"   : 10,
            "cos"   : 10,
            "sin"   : 10,
            "tanh"  : 10,
            "cosh"  : 10,
            "sinh"  : 10,
            "arccos": 10,
            "arcsin": 10,
            "arctan": 10,
        }
    
        self._operators = {
            "power" : '**',
            "divide": '/',
            "times" : '*',
            "minus" : ' - ',
            "plus"  : ' + ',
            "lt"    : ' < ',
            "gt"    : ' > ',
            "leq"   : ' <= ',
            "geq"   : ' >= ',
            "eq"    : ' = ',
            "exp"   : 'exp',
            "ln"    : 'log',
            "abs"   : 'abs',
            "floor" : 'floor',
            "log"   : 'log',
            "root"  : 'sqrt',
            "tan"   : 'tan',
            "cos"   : 'cos',
            "sin"   : 'sin',
            "tanh"  : 'tanh',
            "cosh"  : 'cosh',
            "sinh"  : 'sinh',
            "arccos": 'acos',
            "arcsin": 'asin',
            "arctan": 'atan',
            }

    def use_parenthesis(self, child, parent, first_operand=True):
        """
        Return true if child operation need parenthesis
        """
        if parent is None:
            return False

        parent_prec = self._precedence[parent]
        if parent == "minus" and not first_operand:
            parent_prec -= 0.5

        if parent == "divide" and child == "times" and first_operand:
            return False
        
        if parent == "minus" and child == "plus" and first_operand:
            return False
        
        return parent_prec < self._precedence[child]

    def __getitem__(self, operator):
        return self._operators[operator]
    
    def _gettag(self, node):
        """
        Splits off the namespace part from name, and returns the rest, the tag
        """
        return "".join(node.tag.split("}")[1:])

    def parse(self, root):
        """
        Recursively parse a mathML subtree and return an list of tokens
        together with any state variable and derivative.
        """
        self._state_variable = None
        self._derivative = None
        self.used_variables = set()

        equation_list = self._parse_subtree(root)
        return equation_list, self._state_variable, self._derivative, \
               self.used_variables
    
    def _parse_subtree(self, root, parent=None, first_operand=True):
        op = self._gettag(root)

        # If the tag i "apply" pick the operator and continue parsing
        if op == "apply":
            children = root.getchildren()
            op = self._gettag(children[0])
            root = children[1:]
        # If use special method to parse
        if hasattr(self, "_parse_" + op):
            return getattr(self, "_parse_" + op)(root, parent)
        elif op in self._operators.keys():
            # Build the equation string
            eq  = []
            
            # Check if we need parenthesis
            use_parent = self.use_parenthesis(op, parent, first_operand)
            
            # If unary operator
            if len(root) == 1:
                # Special case if operator is "minus"
                if op == "minus":
                    
                    # If an unary minus is infront of a cn or ci we skip
                    # parenthesize
                    if self._gettag(root[0]) in ["ci", "cn"]:
                        use_parent = False

                    # If an unary minus is infront of a plus we always use parenthesize
                    if self._gettag(root[0]) == "apply" and \
                           self._gettag(root[0].getchildren()[0]) in ["plus"]:
                        use_parent = True

                    eq += ["-"]
                else:
                    
                    # Always use paranthesis for unary operators
                    use_parent = True
                    eq += [self._operators[op]]

                eq += ["("]*use_parent + self._parse_subtree(root[0], op) + \
                      [")"]*use_parent
                return eq
            else:
                # Binary operator
                eq += ["("] * use_parent + self._parse_subtree(root[0], op)
                for operand in root[1:]:
                    eq = eq + [self._operators[op]] + self._parse_subtree(\
                        operand, op, first_operand=False)
                eq = eq + [")"]*use_parent
                return eq
        else:
            error("No support for parsing MathML " + op + " operator.")

    def _parse_conditional(self, condition, operands, parent):
        return [condition] + ["("] + self._parse_subtree(operands[0], parent) \
               + [", "] +  self._parse_subtree(operands[1], parent) + [")"]

    def _parse_and(self, operands, parent):
        ret = ["And("]
        for operand in operands:
            ret += self._parse_subtree(operand, parent) + [", "]
        return ret + [")"]
    
    def _parse_or(self, operands, parent):
        ret = ["Or("]
        for operand in operands:
            ret += self._parse_subtree(operand, parent) + [", "]
        return ret + [")"]
    
    def _parse_lt(self, operands, parent):
        return self._parse_conditional("Lt", operands, "lt")

    def _parse_leq(self, operands, parent):
        return self._parse_conditional("Le", operands, "leq")

    def _parse_gt(self, operands, parent):
        return self._parse_conditional("Gt", operands, "gt")

    def _parse_geq(self, operands, parent):
        return self._parse_conditional("Ge", operands, "geq")

    def _parse_neq(self, operands, parent):
        return self._parse_conditional("Ne", operands, "neq")

    def _parse_eq(self, operands, parent):
        # Parsing conditional
        if parent == "piecewise":
            return self._parse_conditional("Eq", operands, "eq")

        # Parsing assignment
        return self._parse_subtree(operands[0], "eq") + [self["eq"]] + \
               self._parse_subtree(operands[1], "eq")

    def _parse_pi(self, var, parent):
        return ["pi"]
    
    def _parse_ci(self, var, parent):
        varname = var.text.strip()
        if varname in _all_keywords:
            varname = varname + "_"
        self.used_variables.add(varname)
        return [varname]
    
    def _parse_cn(self, var, parent):
        value = var.text.strip()
        if "type" in var.keys() and var.get("type") == "e-notation":
            # Get rid of potential float repr
            exponent = "e" + str(int(var.getchildren()[0].tail.strip()))
        else:
            exponent = ""
        value += exponent
            
        # Fix possible strangeness with integer division in Python...
        nums = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
        num_strs = ["one", "two", "three", "four", "five", "ten"]
        
        if eval(value) in nums:
            value = dict(t for t in zip(nums, num_strs))[eval(value)]
        #elif "." not in value and "e" not in value:
        #    value += ".0"
        
        return [value]
    
    def _parse_diff(self, operands, parent):

        # Store old used_variables so we can erase any collected state
        # variables
        used_variables_prior_parse = self.used_variables.copy()
        
        x = "".join(self._parse_subtree(operands[1], "diff"))
        y = "".join(self._parse_subtree(operands[0], "diff"))
        
        if x in _all_keywords:
            x = x + "_"
        
        d = "d" + x + "d" + y

        # Restore used_variables
        self.used_variables = used_variables_prior_parse

        # Store derivative
        self.used_variables.add(d)
        
        # This is an in/out variable remember it
        self._derivative = d
        self._state_variable = x
        return [d]
    
    def _parse_bvar(self, var, parent):
        if len(var) == 1:
            return self._parse_subtree(var[0], "bvar")
        else:
            error("ERROR: No support for higher order derivatives.")
            
    def _parse_piecewise(self, cases, parent=None):
        if len(cases) == 2:
            piece_children = cases[0].getchildren()
            cond  = self._parse_subtree(piece_children[1], "piecewise")
            true  = self._parse_subtree(piece_children[0])
            false = self._parse_subtree(cases[1].getchildren()[0])
            return ["Conditional", "("] + cond + [", "] + true + [", "] + \
                   false + [")"]
        else:
            piece_children = cases[0].getchildren()
            cond  = self._parse_subtree(piece_children[1], "piecewise")
            true  = self._parse_subtree(piece_children[0])
            return ["Conditional", "("] + cond + [", "] + true + [", "] + \
                   self._parse_piecewise(cases[1:]) + [")"]
    
class MathMLCPPParser(MathMLBaseParser):
    def _parse_power(self, operands):
        return ["pow", "("] + self._parse_subtree(operands[0]) + [", "] + \
               self._parse_subtree(operands[1]) + [")"]

    def _parse_piecewise(self, cases):
        if len(cases) == 2:
            piece_children = cases[0].getchildren()
            cond  = self._parse_subtree(piece_children[1])
            true  = self._parse_subtree(piece_children[0])
            false = self._parse_subtree(cases[1].getchildren()[0])
            return ["("] + cond + ["?"] + true + [":"] + false + [")"]
        else:
            sys.exit("ERROR: No support for cases with other than two "\
                     "possibilities.")

class CellMLParser(object):
    """
    This module parses a CellML XML-file and converts it to PyCC code
    """
    @staticmethod
    def default_parameters():
        return ParameterDict(
            change_state_names=Param([], description="A list of state names "\
                                     "which should be changed to not interfere "\
                                     "with for example a parameter name."), 
            grouping=OptionParam("encapsulation", ["encapsulation", "containment"], \
                                 description="Determines what type of grouping "\
                                 "should be used when the cellml model is parsed."),
            use_sympy_integers=Param(False, description="If yes dedicated sympy "\
                                     "integers will be used instead of the "\
                                     "integers 0-10. This will turn 1/2 into a "\
                                     "sympy rational instead of the float 0.5."),
            )
    
    def __init__(self, model_source, targets=None, params=None):
        """
        Arguments:
        ----------
        
        model_source: str
            Path or url to CellML file
        targets : list (optional)
            Components of the model to parse
        params : dict
            A dict with parameters for the 
        """

        targets = targets or []
        params = params or {}
        check_arg(model_source, str)
        check_arg(targets, list, itemtypes=str)
        self._params = self.default_parameters()
        self._params.update(params)

        # Open file or url
        try:
            fp = open(model_source, "r")
        except IOError:
            try:
                fp = urllib.urlopen(model_source)
            except:
                error("ERROR: Unable to open " + model_source)

        self.model_source = model_source
        self.cellml = ElementTree.parse(fp).getroot()
        self.mathmlparser = MathMLBaseParser()
        self.cellml_namespace = self.cellml.tag.split("}")[0] + "}"
        self.name = self.cellml.attrib['name']

        self.components = self.parse_components(targets)
        self.documentation = self.parse_documentation()

    def parse_documentation(self):
        """
        Parse the documentation of the article
        """
        namespace = "{http://cellml.org/tmp-documentation}"
        article = self.cellml.getiterator(namespace+"article")
        if not article:
            return ""

        article = article[0]

        # Get title
        if article.getiterator(namespace+"articleinfo") and \
               article.getiterator(namespace+"articleinfo")[0].\
               getiterator(namespace+"title"):
            title = article.getiterator(namespace+"articleinfo")[0].\
                    getiterator(namespace+"title")[0].text
        else:
            title = ""

        # Get model structure comments
        for child in article.getchildren():
            if child.attrib.get("id") == "sec_structure":
                content = []
                for par in child.getiterator(namespace+"para"):
                    # Get lines
                    splitted_line = deque(("".join(text.strip() \
                                        for text in par.itertext())).\
                                          split(" "))

                    # Cut them in lines which are not longer than 80 characters
                    ret_lines = []
                    while splitted_line:
                        line_stumps = []
                        line_length = 0 
                        while splitted_line and (line_length + \
                                                 len(splitted_line[0]) < 80):
                            line_stumps.append(splitted_line.popleft())
                            line_length += len(line_stumps[-1]) + 1
                        ret_lines.append(" ".join(line_stumps))
                    
                    content.extend(ret_lines)
                    content.append("\n")

                # Clean up content
                content = ("\n".join(cont.strip() for cont in content)).\
                          replace("  ", " ").replace(" .", ".").replace(\
                    " ,", ",")
                break
        else:
            content = ""
            
        if title or content:
            return "%s\n\n%s" % (title, content)

        return ""


    def _gettag(self, node):
        """
        Splits off the namespace part from name, and returns the rest, the tag
        """
        return "".join(node.tag.split("}")[1:])

    def get_iterator(self, name, item=None):
        """
        Return an element tree iterator 
        """
        
        item = item if item is not None else self.cellml
        return item.getiterator(self.cellml_namespace+name)

    def parse_imported_model(self):
        """
        Parse any imported models
        """

        components = OrderedDict()

        # Collect states and parameters to check for duplicates
        collected_states = dict()
        collected_parameters = dict()

        # Import other models
        for model in self.get_iterator("import"):
            import_comp_names = []
            for comp in self.get_iterator("component", model):
                import_comp_names.append(comp.attrib["name"])
            
            components.update(dict((comp.name, comp) for comp in CellMLParser(\
                model.attrib["{http://www.w3.org/1999/xlink}href"], \
                import_comp_names).components))

        # Extract parameters and states
        for comp in components.values():
            for name in comp.state_variables:
                if name in collected_states:
                    warning("Duplicated state name: '%s' detected in "\
                            "imported component: '%s'" % (name, comp.name))
                collected_states[name] = comp
            
            for name in comp.parameters.keys():
                if name in collected_states + collected_parameters:
                    new_name = name + "_" + comp_name.split("_")[0]
                    comp.change_parameter_name(name, new_name)
                    name = new_name
                collected_parameters[name] = comp

        return components, collected_states, collected_parameters

    def get_parents(self):
        """
        If group was used in the cellml use it to gather parent information
        about the components
        """
        # Collect component encapsulation
        def get_encapsulation(elements, all_parents, parent=None):
            children = {}
            for encap in elements:
                name = encap.attrib["component"]
                all_parents[name] = parent
                if encap.getchildren():
                    nested_children = get_encapsulation(\
                        encap.getchildren(), all_parents, name)
                    children[name] = dict(children=nested_children, \
                                          parent=parent)
                else:
                    children[name] = dict(children=None, parent=parent)

            return children

        all_parents = dict()
        for group in self.get_iterator("group"):
            children = group.getchildren()

            if children and children[0].attrib.get("relationship") == \
                   self._params.grouping:
                encapsulations = get_encapsulation(children[1:], all_parents)

        # If no group information in cellml extract potential parent information
        # from component names
        if not all_parents:

            # Iterate over the components
            comp_names = [comp.attrib["name"] for comp in self.get_iterator(\
                "component")]

            for parent_name in comp_names:
                for name in comp_names:
                    if parent_name in name and parent_name != name:
                        all_parents[name] = parent_name

        return all_parents

    def parse_single_component(self, comp, collected_parameters,
                               collected_states):
        """
        Parse a single component and create a Component object
        """
        comp_name = comp.attrib["name"]
        
        # Collect variables and equations
        variables = OrderedDict()
        equations = []
        state_variables = OrderedDict()
        #derivatives = []

        # Get variable and initial values
        for var in self.get_iterator("variable", comp):

            var_name = var.attrib["name"]
            if var_name in _all_keywords:
                var_name = var_name + "_"

            if var.attrib.has_key("initial_value"):
                initial = var.attrib["initial_value"]
            else:
                initial = None

            # Store variables
            variables[var_name] = initial

        # Get equations
        for math in comp.getiterator(\
            "{http://www.w3.org/1998/Math/MathML}math"):
            for eq in math.getchildren():
                equation_list, state_variable, derivative, \
                        used_variables = self.mathmlparser.parse(eq)

                # Get equation name
                eq_name = equation_list[0]
                
                if eq_name in _all_keywords:
                    equation_list[0] = eq_name + "_"
                    eq_name = equation_list[0]
                
                # Discard collected equation name from used variables
                used_variables.discard(eq_name)
                
                assert(re.findall("(\w+)", eq_name)[0]==eq_name)
                assert(equation_list[1] == self.mathmlparser["eq"])
                equations.append(Equation(eq_name, equation_list[2:],
                                          used_variables))
                
                # Do not register state variables twice
                if state_variable is not None and \
                       state_variable not in state_variables:
                    state_variables[state_variable] = derivative

        # Create and return Component
        comp = Component(comp_name, variables, equations, state_variables)

        # Check for duplicates of parameters and states
        for name in comp.state_variables.keys():
            if name in self._params.change_state_names:
                newname = name + "_" + comp_name.split("_")[0]
                comp.change_state_name(name, newname)
                name = newname
            
            if name in collected_states:
                warning("Duplicated state name: '%s' detected in "\
                        "component: '%s'" % (name, comp.name))
            collected_states[name] = comp

        all_collected_names = collected_states.keys() + \
                              collected_parameters.keys()
        
        for name in comp.parameters.keys():
            if name in all_collected_names:
                new_name = name + "_" + comp_name.split("_")[0]
                if new_name in all_collected_names:
                    new_name = name + "_" + comp_name
                    
                comp.change_parameter_name(name, new_name)
                name = new_name
            collected_parameters[name] = comp
            all_collected_names.append(name)

        return comp

    def add_dependencies_and_sort_components(self, components):
        
        # Check internal dependencies
        for comp0 in components:
            for comp1 in components:
                if comp0 == comp1:
                    continue
                comp0.check_dependencies(comp1)

        def sort_components(components):
            components = deque(components)
            dependant_components = []
            sorted_components = []
            while components:
                component = components.popleft()

                # Chek for circular dependancy
                if dependant_components.count(component) > 4:
                    components.append(component)
                    break
                
                if any(dep in components for dep in \
                       component.components_dependencies):
                    components.append(component)
                    dependant_components.append(component)
                else:
                    sorted_components.append(component)

            return sorted_components, list(components)

        # Initial sorting
        sorted_components, circular_components = sort_components(components)

        # If no circular dependencies
        if not circular_components:
            return sorted_components

        # Collect zero and one dependencies
        zero_dep_equations = set()
        one_dep_equations = set()
        heruistic_score = []
        
        # Gather zero dependent equations
        for comp in circular_components:
            for dep_comp, equations in comp.components_dependencies.items():
                for equation in equations:
                    if not equation.dependent_equations and equation.name \
                           in comp.used_variables:
                        zero_dep_equations.add(equation)

        # Check for one dependency if that is the zero one
        for comp in circular_components:
            for dep_comp, equations in comp.components_dependencies.items():
                for equation in equations:
                    if len(equation.dependent_equations) == 1 and \
                           equation.name in comp.used_variables and \
                           equation.dependent_equations[0] in \
                           zero_dep_equations:
                        one_dep_equations.add(equation)

        # Create heuristic score of which equation we will start removing
        # from components
        for low_dep in list(zero_dep_equations) + list(one_dep_equations):
            num = 0
            for comp in circular_components:
                for equations in comp.components_dependencies.values():
                    num += low_dep in equations
            
            heruistic_score.append((num, low_dep))

        heruistic_score.sort()

        for num, eq, in heruistic_score:
            print num, eq

        # Try to eliminate circular dependency
        # Extract dependent equation to a new component
        ode_comp = Component(self.name, {}, [], {})
        
        removed_equations = {}
        
        # Try to eliminate circular dependency
        while circular_components:
            
            num, eq = heruistic_score.pop()
            
            old_comp = eq.component
            ode_comp.equations.append(eq)

            # Store changed 
            removed_equations[eq] = old_comp

            print
            print "Moving", eq, "from", old_comp

            # Transfer dependent_components to new component
            new_dependent_componets = OrderedDict()
            for dep_comp, equations in old_comp.dependent_components.items():
                assert equations

                if eq not in equations or dep_comp == ode_comp:
                    continue

                print dep_comp,  
                
                # Remove dependency from old component and add it to the new
                if len(equations) == 1:
                    new_dependent_componets[dep_comp] = \
                                    old_comp.dependent_components.pop(dep_comp)
                else:
                    new_dependent_componets[dep_comp] = [\
                        old_comp.dependent_components[dep_comp].pop(equations.index(eq))]

                # Change component dependencies
                if old_comp in dep_comp.components_dependencies:
                    if len(dep_comp.components_dependencies[old_comp]) == 1:
                        dep_comp.components_dependencies.pop(old_comp)
                    else:
                        dep_comp.components_dependencies[old_comp].remove(eq)

                if ode_comp not in dep_comp.components_dependencies:
                    dep_comp.components_dependencies[ode_comp] = [eq]
                else:
                    dep_comp.components_dependencies[ode_comp].append(eq)

            # Store new component to the extracted equation
            eq.component = ode_comp
            ode_comp.dependent_componets = new_dependent_componets
            
            if ode_comp not in sorted_components:
                sorted_components.insert(0, ode_comp)
            
            sorted_components, circular_components = sort_components(\
                sorted_components + circular_components)

        warning("To avoid circular dependency the following equations "\
                "has been moved:")
        
        for eq, old_comp in removed_equations.items():
            warning("{0} : from {1} to {2} component".format(\
                eq.name, old_comp.name, ode_comp.name))

        return components

    def parse_components(self, targets):
        """
        Build a dictionary containing dictionarys describing each
        component of the cellml model
        """

        components, collected_states, collected_parameters = \
                    self.parse_imported_model()

        # Get parent relationship between components
        all_parents = self.get_parents()
            
        # Iterate over the components
        for comp in self.get_iterator("component"):
            comp_name = comp.attrib["name"]

            # Only parse selected and non-empty components
            if (targets and comp_name not in targets) or \
                   len(comp.getchildren()) == 0:
                continue

            # Store component
            components[comp_name] = self.parse_single_component(\
                comp, collected_parameters, collected_states)

        for name, comp in collected_states.items():
            if name in collected_parameters:
                warning("State name: '{0}' in component: '{1}' is a "\
                        "duplication of a parameter in component {2}".format(\
                            name, comp.name, collected_parameters[name].name))
        
        # Add parent information
        for name, comp in components.items():
            parent_name = all_parents.get(name)
            if parent_name:
                comp.parent = components[parent_name]

                # If parent name in child name, reduce child name length
                if parent_name in comp.name:
                    comp.name = comp.name.replace(parent_name, "").strip("_")

        # Add dependencies and sort the components accordingly
        components = self.add_dependencies_and_sort_components(\
            components.values())
        
        return components

    def to_gotran(self):
        """
        Generate a gotran file
        """
        gotran_lines = []
        for docline in self.documentation.split("\n"):
            gotran_lines.append("# " + docline)

        if gotran_lines:
            gotran_lines.extend([""])

        # Add component info
        state_lines = []
        param_lines = []
        equation_lines = []
        derivative_lines = []
        derivatives_intermediates = []

        # Iterate over components and collect stuff
        for comp in self.circular_dependency + self.components:
            comp_name = comp.name.replace("_", " ").capitalize()
            
            # Collect initial state values
            if comp.state_variables:
                state_lines.append("")
                state_lines.append("states(\"{0}\",".format(comp_name))
                for name, value in comp.state_variables.items():
                    state_lines.append("       {0} = {1},".format(name, value))
                state_lines[-1] = state_lines[-1][:-1]+")"

                # Collect derivatives
                for state, derivative in comp.derivatives.items():
                    for eq in comp.equations:
                        if eq.name in derivative:

                            # Check that derivative equation is not used
                            # by component or dependent components
                            for potential_dep in comp.equations + \
                                    comp.dependent_components.keys():
                                if eq.name in potential_dep.used_variables:
                                    derivatives_intermediates.append(eq.name)
                                    derivative_lines.append("d{0}_dt = {1}".\
                                                    format(state, eq.name))
                                    
                                    break
                            else:
                                # Derivative is not used by anyone. Add it to 
                                derivative_lines.append("d{0}_dt = {1}".\
                                            format(state, "".join(eq.expr)))
                            break
            
            # Collect initial parameters values
            if comp.parameters:
                param_lines.append("")
                param_lines.append("parameters(\"{0}\",".format(comp_name))
                for name, value in comp.parameters.items():
                    param_lines.append("           {0} = {1},".format(\
                        name, value))
                param_lines[-1] = param_lines[-1][:-1]+")"

            # Collect all intermediate equations
            if comp.equations:
                equation_lines.append("")
                if comp in self.circular_dependency:
                    equation_lines.append("# Equations with circular "\
                                          "dependency")
                equation_lines.append("component(\"{0}\")".format(\
                    comp_name))
                
                for eq in comp.equations:
                    
                    # If derivative line and not used else where continue
                    if eq.name in comp.derivatives.values() and \
                           eq.name not in derivatives_intermediates:
                        continue
                    equation_lines.append("{0} = {1}".format(eq.name, \
                                                             "".join(eq.expr)))

        # FIXME: Add logic for DAE

        gotran_lines.append("# gotran file generated by cellml2gotran from "\
                            "{0}".format(self.model_source))
        gotran_lines.extend(state_lines)
        gotran_lines.extend(param_lines)
        gotran_lines.extend(equation_lines)
        gotran_lines.append("")
        gotran_lines.append("comment(\"The ODE system: {0} states\")".format(\
            len(derivative_lines)))
        gotran_lines.extend(derivative_lines)
        gotran_lines.append("")
        

        # Return joined lines
        return "\n".join(gotran_lines)

        # Write file
        open("{0}.ode".format(self.name), \
             "w").write()

def cellml2ode(model_source, extract_equations=None, change_state_names=None):
    """
    Convert a CellML model into an ode

    Arguments:
    ----------
    model_source: str
        Path or url to CellML file
    extract_equations : list of str (optional)
        List of equations which should be extracted from its component
        to prevent circular dependency
    change_state_names : list of str
        List of str with state names which should have it name locally
        changed
    """
    from gotran import exec_ode
    cellml = CellMLParser(model_source, extract_equations=extract_equations, \
                          change_state_names=change_state_names)
    return exec_ode(cellml.to_gotran(), cellml.name)
    
