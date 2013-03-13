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
from gotran.common import warning, error

__all__ = ["cellml2ode", "CellMLParser"]

ui = "UNINITIALIZED"

python_keywords = ["and", "del", "from", "not", "while", "as", "elif",
                   "global", "or", "with", "assert", "else", "if", "pass",
                   "yield", "break", "except", "import", "print", "class",
                   "exec", "in", "raise", "continue", "finally", "is",
                   "return", "def", "for", "lambda", "try"]

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
            "and"   : 8,
            "or"    : 9,
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
            "and"   : ' & ',
            "or"    : ' | ',
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

    def use_parenthesis(self, child, parent):
        """
        Return true if child operation need parenthesis
        """
        if parent is None:
            return False

        parent_prec = self._precedence[parent]
        if parent == "minus":
            parent_prec -= 0.5
        
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
    
    def _parse_subtree(self, root, parent=None):
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
            use_parent = self.use_parenthesis(op, parent)
            
            # If unary operator
            if len(root) == 1:
                # Special case if operator is "minus"
                if op == "minus":
                    
                    # If an unary minus is infront of a cn or ci we skip
                    # parenthesize
                    if self._gettag(root[0]) in ["ci", "cn"]:
                        use_parent = False

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
                        operand, op)
                eq = eq + [")"]*use_parent
                return eq
        else:
            error("No support for parsing MathML " + op + " operator.")

    def _parse_conditional(self, condition, operands, parent):
        return [condition] + ["("] + self._parse_subtree(operands[0], parent) \
               + [", "] +  self._parse_subtree(operands[1], parent) + [")"]
    
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
        if varname in python_keywords:
            varname = varname + "_"
        self.used_variables.add(varname)
        return [varname]
    
    def _parse_cn(self, var, parent):
        value = var.text.strip()
        # Fix possible strangeness with integer division in Python...
        nums = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
        num_strs = ["one", "two", "three", "four", "five", "ten"]
        
        if eval(value) in nums:
            value = dict(t for t in zip(nums, num_strs))[eval(value)]
        elif parent == "divide" and "." not in value:
            value += ".0"
        return [value]
    
    def _parse_diff(self, operands, parent):

        # Store old used_variables so we can erase any collected state
        # variables
        used_variables_prior_parse = self.used_variables.copy()
        
        x = "".join(self._parse_subtree(operands[1], "diff"))
        y = "".join(self._parse_subtree(operands[0], "diff"))
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
    def __init__(self, model_source, targets=None, extract_equations=None, \
                 change_state_names=None):
        """
        Arguments:
        ----------
        
        model_source: str
            Path or url to CellML file
        targets : list (optional)
            Components of the model to parse
        extract_equations : list of str (optional)
            List of equations which should be extracted from its component
            to prevent circular dependency
        change_state_names : list of str
            List of str with state names which should have it name locally
            changed
        """

        # Open file or url
        try:
            fp = open(model_source, "r")
        except IOError:
            try:
                fp = urllib.urlopen(model_source)
            except:
                error("ERROR: Unable to open " + model_source)

        extract_equations = extract_equations or []
        change_state_names = change_state_names or []

        self.model_source = model_source
        self.cellml = ElementTree.parse(fp).getroot()
        self.mathmlparser = MathMLBaseParser()
        self.name = self.cellml.attrib['name']

        self.components, self.circular_dependency, self.zero_dep_equations, \
                         self.one_dep_equations, self.heuristic_score  = \
                         self.parse_components(targets, extract_equations, \
                                               change_state_names)
        self.documentation = self.parse_documentation()

        #if self.circular_dependency:
        #    raise RuntimeError("Circular dependency detected. ")
        
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

    def parse_components(self, targets, extract_equations, change_state_names):
        """
        Build a dictionary containing dictionarys describing each
        component of the cellml model
        """
        components = deque()
        cellml_namespace = self.cellml.tag.split("}")[0] + "}"

        # Collect states and parameters to check for duplicates
        collected_states = dict()
        collected_parameters = dict()
        
        # Import other models
        for model in self.cellml.getiterator(cellml_namespace + "import"):
            import_comp_names = []
            for comp in model.getiterator(cellml_namespace + "component"):
                import_comp_names.append(comp.attrib["name"])
            
            components.extend(CellMLParser(\
                model.attrib["{http://www.w3.org/1999/xlink}href"], \
                import_comp_names).components)

        # Extract parameters and states
        for comp in components:
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
            
        extracted_equations = []
        
        # Iterate over the components
        for comp in self.cellml.getiterator(cellml_namespace + "component"):
            comp_name = comp.attrib["name"]

            # Only parse selected and non-empty components
            if (targets and comp_name not in targets) or \
                   len(comp.getchildren())== 0:
                continue

            # Collect variables and equations
            variables = OrderedDict()
            equations = []
            state_variables = OrderedDict()
            #derivatives = []

            # Get variable and initial values
            for var in comp.getiterator(cellml_namespace + "variable"):

                var_name = var.attrib["name"]
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
                    
                    if eq_name in python_keywords:
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

            # Extract any equations labled for extraction
            if extract_equations:
                for equation in equations[:]:
                    if equation.name in extract_equations:
                        extracted_equations.append(equation)
                        equations.remove(equation)

            # Create Component
            comp = Component(comp_name, variables, equations, state_variables)

            # Check for duplicates of parameters and states
            for name in comp.state_variables.keys():
                if name in change_state_names:
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
            
            # Store component
            components.append(comp)

        for name, comp in collected_states.items():
            if name in collected_parameters:
                warning("State name: '{0}' in component: '{1}' is a "\
                        "duplication of a parameter in component {2}".format(\
                            name, comp.name, collected_parameters[name].name))

        # Add extracted equations as a new Component
        if extracted_equations:
            components.append(Component("Extracted_equations", {}, \
                                        extracted_equations, {}))
        elif extract_equations:
            warning("No equations were extracted.")
        
        # Check internal dependencies
        for comp0 in components:
            #all_equations.extend(comp0.equations)
            for comp1 in components:
                if comp0 == comp1:
                    continue
                comp0.check_dependencies(comp1)

        def sort_components(components):
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
        sorted_components, circular_components = sort_components(\
            deque(components))

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

        heruistic_score.sort(reverse=True)

        if heruistic_score:
            warning("Circular dependency detetected. A heuristic score of "\
                    "which equations it might be benefitial to extract "\
                    "follows:")
            for num, dep in heruistic_score:
                warning("{0} : {1}".format(num, dep))

        return sorted_components, circular_components, \
               list(zero_dep_equations), list(one_dep_equations), \
               heruistic_score

        # Try to eliminate circular dependency
        while circular_components:
            
            zero_dep_equations = []
            
            # Gather zero dependent equations
            for comp in circular_components:
                for dep_comp, equations in \
                        comp.components_dependencies.items():
                    for equation in equations:
                        if not equation.dependent_equations:
                            zero_dep_equations.append(equation)
            
            sorted_components, circular_components = \
                        sort_components(sorted_components+circular_components)

        return sorted_components, circular_components


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
    
