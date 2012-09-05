#!/usr/bin/env python

import sys, os, re
import urllib
from xml.etree import ElementTree

from collections import OrderedDict, deque

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
        return "Equation({0} = {1})".format(self.name, self.expr)

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
    def __init__(self, name, variables, equations, state_variables=None,
                 derivatives=None):
        self.name = name

        self.state_variables = OrderedDict((state, variables.pop(state, None)) \
                                           for state in state_variables)

        self.parameters = OrderedDict((name, value) for name, value in \
                                      variables.items() if value is not None)
        
        self.derivatives = derivatives
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

        self.used_variables.difference_update(\
            equation.name for equation in self.equations)

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name + "<{0}>".format(len(self.state_variables))

    def __repr__(self):
        return "Component<{0}, {1}>".format(self.name, len(self.state_variables))

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
            self.dependent_components[component] = \
                [equation for equation in component.equations \
                 if equation.name in self.used_variables]
            
            # Add logics for dependencies of all Equations.
            for other_equation in component.equations:
                for equation in self.equations:
                    if other_equation.name in equation.used_variables:
                        equation.dependent_equations.append(other_equation)
            

class MathMLBaseParser(object):
    def __init__(self):
        self._state_variable = None
        self._derivative = None
        self.variables_names = set()
    
        self._precedence = {
            "power" : 0,
            "divide": 1,
            "times" : 1,
            "minus" : 4,
            "plus"  : 5,
            "lt"    : 6,
            "gt"    : 6,
            "leq"   : 6,
            "geq"   : 6,
            "and"   : 8,
            "xor"   : 8,
            "or"    : 8,
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
            "times" : ' * ',
            "minus" : ' - ',
            "plus"  : ' + ',
            "lt"    : ' < ',
            "gt"    : ' > ',
            "leq"   : ' <= ',
            "geq"   : ' >= ',
            "and"   : ' and ',
            "xor"   : ' ?? ',
            "or"    : ' || ',
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
        return equation_list, self._state_variable, self._derivative, self.used_variables
    
    def _parse_subtree(self, root, parent = None):
        op = self._gettag(root)

        # If the tag i "apply" pick the operator and continue parsing
        if op == "apply":
            children = root.getchildren()
            op = self._gettag(children[0])
            root = children[1:]
        # If use special method to parse
        if hasattr(self, "_parse_" + op):
            return getattr(self, "_parse_" + op)(root)
        elif op in self._operators.keys():
            # Build the equation string
            eq  = []
            
            # Check if we need parentesis
            use_parent = (not parent is None) and \
                         (self._precedence[parent] < self._precedence[op])
            
            # If unary operator
            if len(root) == 1:
                # Special case if operator is "minus"
                if op == "minus":
                    use_parent = False
                    eq += ["-"]
                else:    
                    eq += [self._operators[op]]
                eq += ["("]*use_parent + self._parse_subtree(root[0],op) + [")"]*use_parent
                return eq
            else:
                # Binary operator
                eq += ["("] * use_parent + self._parse_subtree(root[0],op)
                for operand in root[1:]:
                    eq = eq + [self._operators[op]] + self._parse_subtree(operand,op)
                eq = eq + [")"]*use_parent
                return eq
        else:
            raise AttributeError,"No support for parsing MathML " + op + " operator."
    
    def _parse_ci(self, var):
        var = var.text.strip()
        self.used_variables.add(var)
        return [var]
    
    def _parse_cn(self, var):
        #print "CN", var.text.strip()
        return [var.text.strip()]
    
    def _parse_diff(self, operands):
        x = "".join(self._parse_subtree(operands[1]))
        y = "".join(self._parse_subtree(operands[0]))
        d = "d" + x + "d" + y
        
        # This is an in/out variable remember it
        self._derivative = d
        self._state_variable = x
        return [d]
    
    def _parse_bvar(self, var):
        if len(var) == 1:
            return self._parse_subtree(var[0])
        else:
            sys.exit("ERROR: No support for higher order derivatives.")                
            
    def _parse_piecewise(self, cases):
        if len(cases) == 2:
            piece_children = cases[0].getchildren()
            cond  = self._parse_subtree(piece_children[1])
            true  = self._parse_subtree(piece_children[0])
            false = self._parse_subtree(cases[1].getchildren()[0])
            return ["Conditional", "("] + cond + [", "] + true + [", "] + false + [")"]
        else:
            piece_children = cases[0].getchildren()
            cond  = self._parse_subtree(piece_children[1])
            true  = self._parse_subtree(piece_children[0])
            return ["Conditional", "("] + cond + [", "] + true + [", "] + \
                   self._parse_piecewise(cases[1:]) + [")"]
    
    def _parse_eq(self, operands):
        return self._parse_subtree(operands[0]) + [self["eq"]] + \
               self._parse_subtree(operands[1])

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
            sys.exit("ERROR: No support for cases with other than two possibilities.")

class CellMLParser:
    """
    This module parses a CellML XML-file and converts it to PyCC code
    """
    def __init__(self, model_source, targets=None, extract_equations=None):
        """
        model_source: path or url to CellML file
        targets:      the components of the model to parse
        modelname:    name of the model
        """

        # Open file or url
        try:
            fp = open(model_source, "r")
        except IOError:
            try:
                fp = urllib.urlopen(model_source)
            except:
                raise RuntimeError("ERROR: Unable to open " + model_source)

        extract_equations = extract_equations or []
        
        self.cellml = ElementTree.parse(fp).getroot()
        self.mathmlparser = MathMLBaseParser()
        self.name = self.cellml.attrib['name']

        self.components, self.circular_dependency, self.zero_dep_equations, \
                         self.one_dep_equations, self.heuristic_score  = \
                         self.parse_components(targets, extract_equations)
        self.documentation = self.parse_documentation()

        if self.circular_dependency:
            raise RuntimeError("Circular dependency detected. ")
        
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
                                        for text in par.itertext())).split(" "))

                    # Cut them in lines which are not longer than 80 characters
                    ret_lines = []
                    while splitted_line:
                        line_stumps = []
                        line_length = 0 
                        while splitted_line and (line_length + len(splitted_line[0]) < 80):
                            line_stumps.append(splitted_line.popleft())
                            line_length += len(line_stumps[-1]) + 1
                        ret_lines.append(" ".join(line_stumps))
                    
                    content.extend(ret_lines)
                    content.append("\n")

                # Clean up content
                content = ("\n".join(cont.strip() for cont in content)).\
                          replace("  ", " ").replace(" .", ".").replace(" ,", ",")
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

    def parse_components(self, targets, extract_equations):
        """
        Build a dictionary containing dictionarys describing each
        component of the cellml model
        """
        components = deque()
        cellml_namespace = self.cellml.tag.split("}")[0] + "}"
        
        # Import other models
        for model in self.cellml.getiterator(cellml_namespace + "import"):
            import_comp_names = []
            for comp in model.getiterator(cellml_namespace + "component"):
                import_comp_names.append(comp.attrib["name"])
            
            components.extend(CellMLParser(\
                model.attrib["{http://www.w3.org/1999/xlink}href"], \
                import_comp_names).components)
            
        extracted_equations = []
        
        # Iterate over the components
        for comp in self.cellml.getiterator(cellml_namespace + "component"):
            comp_name = comp.attrib["name"]

            # Only parse selected and non-empty components
            if (targets and comp_name not in targets) or len(comp.getchildren())== 0:
                continue

            # Collect variables and equations
            variables = OrderedDict()
            equations = []
            state_variables = []
            derivatives = []

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

                    eq_name = equation_list[0]
                    assert(re.findall("(\w+)", eq_name)[0]==eq_name)
                    assert(equation_list[1] == self.mathmlparser["eq"])
                    equations.append(Equation(eq_name, "".join(equation_list[2:]),
                                              used_variables))
                    
                    if not state_variable is None:
                        state_variables.append(state_variable)
                    
                    if not derivative is None:
                        derivatives.append(derivative)

            # FIXME: Add logic to extract variables
            if extract_equations:
                for equation in equations[:]:
                    if equation.name in extract_equations:
                        extracted_equations.append(equation)
                        equations.remove(equation)
            
            # Make a list of the used_variables
            components.append(Component(comp_name, variables, \
                                        equations, state_variables, derivatives))

        # Add extracted equations as a new Component
        if extracted_equations:
            components.append(Component("ExtractedEquations", [], \
                                        extracted_equations, [], []))
        
        #all_equations = []
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
                
                if any(dep in components for dep in component.dependent_components):
                    components.append(component)
                    dependant_components.append(component)
                else:
                    sorted_components.append(component)

            return sorted_components, list(components)

        # Initial sorting
        sorted_components, circular_components = sort_components(deque(components))

        zero_dep_equations = set()
        one_dep_equations = set()
        heruistic_score = []
        
        # Gather zero dependent equations
        for comp in circular_components:
            for dep_comp, equations in comp.dependent_components.items():
                for equation in equations:
                    if not equation.dependent_equations and equation.name \
                           in comp.used_variables:
                        zero_dep_equations.add(equation)

        # Check for one dependency if that is the zero one
        for comp in circular_components:
            for dep_comp, equations in comp.dependent_components.items():
                for equation in equations:
                    if len(equation.dependent_equations) == 1 and \
                           equation.name in comp.used_variables and \
                           equation.dependent_equations[0] in zero_dep_equations:
                        one_dep_equations.add(equation)

        # Create heuristic score of which equation we will start removing
        # from components
        for low_dep in list(zero_dep_equations) + list(one_dep_equations):
            num = 0
            for comp in circular_components:
                for equations in comp.dependent_components.values():
                    num += low_dep in equations
            
            heruistic_score.append((num, low_dep))

        heruistic_score.sort()

        print heruistic_score

        return sorted_components, circular_components, list(zero_dep_equations), \
               list(one_dep_equations), heruistic_score

        # Try to eliminate circular dependency
        while circular_components:
            
            zero_dep_equations = []
            
            # Gather zero dependent equations
            for comp in circular_components:
                for dep_comp, equations in comp.dependent_components.items():
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

        # Iterate over components and collect stuff
        for comp in self.components:
            comp_name = comp.name.replace("_", " ").capitalize()
            if comp.state_variables:
                state_lines.append("")
                state_lines.append("states(\"{0}\",".format(comp_name))
                for name, value in comp.state_variables.items():
                    state_lines.append("       {0}={1},".format(name, value))
                state_lines[-1] = state_lines[-1][:-1]+")"

                # Collect derivatives
                for state, derivative in zip(comp.state_variables, \
                                             comp.derivatives):
                    for eq in comp.equations:
                        if eq.name in derivative:
                            derivative_lines.append("diff({0}, {1})".format(\
                                state, eq.expr))
                            break
                                   
            if comp.parameters:
                param_lines.append("")
                param_lines.append("parameters(\"{0}\",".format(comp_name))
                for name, value in comp.parameters.items():
                    param_lines.append("           {0}={1},".format(name, value))
                param_lines[-1] = param_lines[-1][:-1]+")"

            if comp.equations:
                equation_lines.append("")
                equation_lines.append("comment(\"{0}\")".format(\
                    comp_name))
                
                for eq in comp.equations:
                    
                    # If derivative line continue
                    if eq.name in comp.derivatives:
                        continue
                    equation_lines.append("{0}={1}".format(eq.name, eq.expr))

        # FIXME: Add logic for DAE

        gotran_lines.append("# gotran file generated by cellml2gotran from ")
        gotran_lines.extend(state_lines)
        gotran_lines.extend(param_lines)
        gotran_lines.extend(equation_lines)
        gotran_lines.append("")
        gotran_lines.append("comment(\"The ODE system: {0} states\")".format(\
            len(derivative_lines)))
        gotran_lines.extend(derivative_lines)
        gotran_lines.append("")

        # Write file
        open("{0}.ode".format(self.name), \
             "w").write("\n".join(gotran_lines))
                                   
if __name__ == "__main__":
    import getopt

    # Filepointers and fallback values
    c_out_fp = sys.stdout
    h_out_fp = sys.stdout
    c_skel_fp = None
    h_skel_fp = None

    #        Flag    Description             Variable   Option
    flags = {"co" : ["cpp output file",      c_out_fp,  "w"],
             "ho" : ["header output file",   h_out_fp,  "w"],
             "cs" : ["cpp skeleton file",    c_skel_fp, "r"],
             "hs" : ["header skeleton file", h_skel_fp, "r"],
             "m"  : ["model name",           None,      "MODELNAME"]}

    # Print usage and exit
    def usage():
        sys.stderr.write("Usage: %s %s %s\n" % (sys.argv[0], " ".join(["[--" + k + "=" + flags[k][0] + "]" for k in flags]), "file | url"))
        sys.exit(-1)
        
    # Parse command line options
    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], "", [elm + "=" for elm in flags.keys()])
        if len(args) != 1:
            usage()
    except getopt.GetoptError:
        usage()

    header_name = "##HEADERNAME##"
    for k,i in opts:
        key = k[2:]
        if key in flags:
            if key == "m":
                flags[key][2] = i
            else:
                # Open selected files
                flags[key][1] = open(i, flags[key][2])
            if key == "ho":
                header_name = i
    
    cellml = CellMLParser(args[0], modelname = flags["m"][2])
    cellml.to_gotran()
