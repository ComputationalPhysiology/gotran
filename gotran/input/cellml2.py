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

from collections import OrderedDict, deque, defaultdict
from gotran.common import warning, error, check_arg

from modelparameters.codegeneration import _all_keywords
from modelparameters.parameterdict import *

from mathml import MathMLBaseParser

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
        self.children = []
        
        self.derivatives = state_variables

        self.components_dependencies = OrderedDict() 
        self.dependent_components = OrderedDict() 

        # Attributes which will be populated later
        # FIXME: Should we populate the parameters based on the variables
        # FIXME: with initial values which are not state variables
        self.sort_and_store_equations(equations)

        # Get used variables
        self.used_variables = set()
        for equation in self.equations:
            self.used_variables.update(equation.used_variables)

        # Remove dependencies on names defined by component
        #self.used_variables.difference_update(\
        #    equation.name for equation in self.equations)

        self.used_variables.difference_update(\
            name for name in self.parameters)

        self.used_variables.difference_update(\
            name for name in self.state_variables)

    def sort_and_store_equations(self, equations):
        
        # Check internal dependencies
        for eq0 in equations:
            
            eq0.dependent_equations = []
            
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

    def change_equation_name(self, oldname, newname):
        
        warning("Locally change equation name: '{0}' to '{1}' in "\
                "component '{2}'.".format(oldname, newname, self.name))
        
        # Update equations
        new_equations = []
        for eqn in self.equations:
            while oldname in eqn.expr:
                eqn.expr[eqn.expr.index(oldname)] = newname

            if eqn.name == oldname:
                eqn.name = newname

            new_equations.append(eqn)

        self.equations = new_equations

    def change_variable_name(self, oldname, newname):
        vartype = self.get_variable_type(oldname)

        if vartype is None:
            error("Cannot change variable name. {0} is not a variable "\
                  "in component {1}".format(oldname, self.name))
            
        if vartype == "state":
            self.change_state_name(oldname, newname)
        elif vartype == "parameter":
            self.change_parameter_name(oldname, newname)
        else:
            self.change_equation_name(oldname, newname)

    def get_variable_type(self, variable):
        if variable in self.state_variables:
            return "state"
        elif variable in self.parameters:
            return "parameter"
        else:
            for eq in self.equations:
                if eq.name == variable:
                    break
            else:
                return
            return "equation"

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
        targets : list, dict (optional)
            Components of the model to parse
        params : dict
            A dict with parameters for the 
        """

        targets = targets or []
        params = params or {}
        check_arg(model_source, str)
        check_arg(targets, (list, dict))
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
        self.mathmlparser = MathMLBaseParser(self._params.use_sympy_integers)
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
        collected_equations = dict()

        # Import other models
        for model in self.get_iterator("import"):
            import_comp_names = dict()

            for comp in self.get_iterator("component", model):
                
                import_comp_names[comp.attrib["component_ref"]] = \
                                                        comp.attrib["name"]

            model_parser = CellMLParser(\
                model.attrib["{http://www.w3.org/1999/xlink}href"], \
                import_comp_names)

            for comp in model_parser.components:
                components[comp.name] = comp

        # Extract parameters and states
        for comp in components.values():

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
            
            for eq in comp.equations:
                collected_equations[eq.name] = comp
            
            all_collected_names = collected_states.keys() + \
                                  collected_parameters.keys() + \
                                  collected_equations.keys()
            
            for name in comp.parameters.keys():
                if name in all_collected_names:
                    new_name = name + "_" + comp.name.split("_")[0]
                    if new_name in all_collected_names:
                        new_name = name + "_" + comp.name
                        
                    comp.change_parameter_name(name, new_name)
                    name = new_name
                collected_parameters[name] = comp
                all_collected_names.append(name)

        return components, collected_states, collected_parameters,\
               collected_equations

    def get_parents(self, grouping, element=None):
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

        encapsulations = dict()
        all_parents = dict()
        for group in self.get_iterator("group", element):
            children = group.getchildren()

            if children and children[0].attrib.get("relationship") == \
                   grouping:
                encapsulations = get_encapsulation(children[1:], all_parents)

        # If no group information in cellml extract potential parent information
        # from component names
        if not all_parents:

            # Iterate over the components
            comp_names = [comp.attrib["name"] for comp in self.get_iterator(\
                "component", element)]

            for parent_name in comp_names:
                for name in comp_names:
                    if parent_name in name and parent_name != name:
                        all_parents[name] = parent_name

        return encapsulations, all_parents

    def parse_single_component(self, comp, collected_parameters,
                               collected_states, collected_equations):
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

        for eq in comp.equations:
            collected_equations[eq.name] = comp

        all_collected_names = collected_states.keys() + \
                              collected_parameters.keys() + \
                              collected_equations.keys()
        
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
        for low_dep in zero_dep_equations:
            num = 0
            for comp in circular_components:
                for equations in comp.components_dependencies.values():
                    num += low_dep in equations
            
            heruistic_score.append((1, num, low_dep))

        for low_dep in one_dep_equations:
            num = 0
            for comp in circular_components:
                for equations in comp.components_dependencies.values():
                    num += low_dep in equations
            
            heruistic_score.append((0, num, low_dep))

        heruistic_score.sort()

        print "Heuristic"
        for deps, num, eq in heruistic_score:
            print num, eq

        # Try to eliminate circular dependency
        # Extract dependent equation to a new component
        ode_comp = Component(self.name, {}, [], {})
        
        removed_equations = {}
        
        # Try to eliminate circular dependency
        while circular_components:
            
            deps, num, eq = heruistic_score.pop()
            
            old_comp = eq.component
            ode_comp.equations.append(eq)
            old_comp.equations.remove(eq)

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

                # Remove dependency from old component and add it to the new
                if len(equations) == 1:
                    new_dependent_componets[dep_comp] = \
                                    old_comp.dependent_components.pop(dep_comp)
                else:
                    new_dependent_componets[dep_comp] = [\
                        old_comp.dependent_components[dep_comp].pop(equations.index(eq))]

                # Change component dependencies
                if old_comp in dep_comp.components_dependencies and eq in \
                       dep_comp.components_dependencies[old_comp]:
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

        # Sort newly added equations
        ode_comp.sort_and_store_equations(ode_comp.equations)

        warning("To avoid circular dependency the following equations "\
                "has been moved:")
        
        for eq, old_comp in removed_equations.items():
            warning("{0} : from {1} to {2} component".format(\
                eq.name, old_comp.name, ode_comp.name))

        return sorted_components

    def parse_components(self, targets):
        """
        Build a dictionary containing dictionarys describing each
        component of the cellml model
        """

        # Parse imported components
        components, collected_states, collected_parameters, \
                    collected_equations = self.parse_imported_model()

        # Get parent relationship between components
        encapsulations, all_parents = self.get_parents(self._params.grouping)

        if targets:

            # If the parent information was not of type encapsulation
            # regather parent information
            if self._params.grouping != "encapsulation":

                encapsulations, dummy = self.get_parents("encapsulation")

            # Add any encapsulated components to the target list
            for target, new_target_name in targets.items():
                if target in encapsulations:
                    for child in encapsulations[target]["children"]:
                        targets[child] = child.replace(\
                            target, new_target_name)

            target_parents = dict()

            # Update all_parents
            for comp_name, parent_name in all_parents.items():
                if parent_name not in targets:
                    continue
                target_parents[targets[comp_name]] = targets[parent_name]
        
        # Iterate over the components
        for comp in self.get_iterator("component"):
            comp_name = comp.attrib["name"]

            # Only parse selected and non-empty components
            if (targets and comp_name not in targets) or \
                   len(comp.getchildren()) == 0:
                continue

            # If targets provides a name mapping give the component a new name
            if targets and isinstance(targets, dict):
                new_name = targets[comp_name]
                comp.attrib["name"] = new_name
                comp_name = new_name
            
            # Store component
            components[comp_name] = self.parse_single_component(\
                comp, collected_parameters, collected_states, collected_equations)

        for name, comp in collected_parameters.items():
            if name in collected_states:
                warning("Parameter name: '{0}' in component: '{1}' is a "\
                        "duplication of a state in component {2}".format(\
                            name, comp.name, collected_states[name].name))
                new_name = name + "_" + comp.name.split("_")[0]
                comp.change_parameter_name(name, new_name)

            elif name in collected_equations:
                warning("Parameter name: '{0}' in component: '{1}' is a "\
                        "duplication of a equation in component {2}".format(\
                            name, comp.name, collected_equations[name].name))
                new_name = name + "_" + comp.name.split("_")[0]
                comp.change_parameter_name(name, new_name)
            
        # Add parent information
        for name, comp in components.items():

            if targets:
                parent_name = target_parents.get(name)
            else:
                parent_name = all_parents.get(name)

            if parent_name:
                comp.parent = components[parent_name]
                
                # If parent name in child name, reduce child name length
                if parent_name in comp.name:
                    comp.name = comp.name.replace(parent_name, "").strip("_")

                components[parent_name].children.append(comp)

        # If we only extract a sub set of component we do not sort
        if targets:
            return components.values()

        # Before dependencies are checked we change names according to
        # variable mappings in the original CellML file
        new_variable_names, same_variable_names = self.parse_name_mappings()
        for comp, variables in new_variable_names.items():

            # Iterate over old and new names
            for oldname, newnames in variables.items():
                
                # Check if the oldname is used in any components
                oldname_used = oldname in same_variable_names[comp]
                
                # If there are only one newname we change the name of the
                # original equation
                if len(newnames) == 1:

                    if not oldname_used:
                        newname = newnames.keys()[0]
                        if components[comp].get_variable_type(oldname) is not None:
                            components[comp].change_variable_name(oldname, newname)
                        else:
                            for child in components[comp].children:
                                if child.get_variable_type(oldname) is not None:
                                    child.change_variable_name(oldname, newname)
                                    break
                    else:
                        # FIXME: Add equation with name change to component
                        pass
                else:
                    # FIXME: Add equation with name change to component
                    pass
        
        # Add dependencies and sort the components accordingly
        components = self.add_dependencies_and_sort_components(\
            components.values())
        
        return components


    def parse_name_mappings(self):
        new_variable_names = dict()
        same_variable_names = dict()
        
        for con in self.get_iterator("connection"):
            con_map = self.get_iterator("map_components", con)[0]
            comp1 = con_map.attrib["component_1"]
            comp2 = con_map.attrib["component_2"]
            
            for var_map in self.get_iterator("map_variables", con):
                var1 = var_map.attrib["variable_1"]
                var2 = var_map.attrib["variable_2"]
                
                if var1 != var2:
                    if comp1 not in new_variable_names:
                        new_variable_names[comp1] = {var1:defaultdict(list)}
                    elif var1 not in new_variable_names[comp1]:
                        new_variable_names[comp1][var1] = defaultdict(list)
                    new_variable_names[comp1][var1][var2].append(comp2)

                else:
                    if comp1 not in same_variable_names:
                        same_variable_names[comp1] = {var1:[comp2]}
                    elif var1 not in same_variable_names[comp1]:
                        same_variable_names[comp1][var1] = [comp2]
                    else:
                        same_variable_names[comp1][var1].append(comp2)

        return new_variable_names, same_variable_names
    
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
        declaration_lines = []
        equation_lines = []

        def unders_score_replace(comp):

            new_name = comp.name.replace("_", " ")

            # If only 1 state it might be included in the name
            if len(comp.state_variables) == 1:
                state_name = comp.state_variables.keys()[0]
                if state_name in comp.name:
                    new_name = state_name.join(part.replace("_", " ") \
                                for part in comp.name.split(state_name))

            single_words = new_name.split(" ")
            if len(single_words[0]) > 1 and "_" not in single_words[0]:
                single_words[0] = single_words[0][0].upper()+single_words[0][1:]

            return " ".join(single_words)

        # Iterate over components and collect stuff
        for comp in self.components:
            
            names = deque([unders_score_replace(comp)])

            parent = comp.parent
            while parent is not None:
                names.appendleft(unders_score_replace(parent))
                parent = parent.parent

            comp_name = ", ".join("\"{0}\"".format(name) for name in names)
            
            # Collect initial state values
            if comp.state_variables:
                declaration_lines.append("")
                declaration_lines.append("states({0} ,".format(comp_name))
                for name, value in comp.state_variables.items():
                    declaration_lines.append("       {0} = {1},".format(name, value))
                declaration_lines[-1] = declaration_lines[-1][:-1]+")"

            # Collect initial parameters values
            if comp.parameters:
                declaration_lines.append("")
                declaration_lines.append("parameters({0} ,".format(comp_name))
                for name, value in comp.parameters.items():
                    declaration_lines.append("           {0} = {1},".format(\
                        name, value))
                declaration_lines[-1] = declaration_lines[-1][:-1]+")"

            # Collect all intermediate equations
            if comp.equations:
                equation_lines.append("")
                equation_lines.append("component({0})".format(\
                    comp_name))

                equation_lines.extend("{0} = {1}".format(eq.name, "".join(eq.expr))\
                                      for eq in comp.equations)

        gotran_lines.append("# gotran file generated by cellml2gotran from "\
                            "{0}".format(self.model_source))
        gotran_lines.extend(declaration_lines)
        gotran_lines.extend(equation_lines)
        gotran_lines.append("")
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
    
