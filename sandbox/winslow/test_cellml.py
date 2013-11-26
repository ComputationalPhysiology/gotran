from collections import defaultdict, OrderedDict
from gotran.input.cellml2 import CellMLParser
import re

model = "winslow_rice_jafri_marban_ororke_1999.cellml"
#model = "iyer_mazhari_winslow_2004.cellml"
#model = "severi_fantini_charawi_difrancesco_2012.cellml"
#model = "terkildsen_niederer_crampin_hunter_smith_2008.cellml"
model = "Pandit_Hinch_Niederer.cellml"

extract_equations = []
change_state_names = []

params = CellMLParser.default_parameters()

cellml = CellMLParser(model, params=params)

mathmlparser = cellml.mathmlparser
parsed_components = cellml.components
cellml = cellml.cellml

cellml_namespace = cellml.tag.split("}")[0] + "}"

# Grab encapsulation grouping
encapsulations = {}

def get_encapsulation(elements):

    children = {}
    for encap in elements:
        name = encap.attrib["component"]
        if encap.getchildren():
            children[name] = get_encapsulation(encap.getchildren())
        else:
            children[name] = None
    return children

for group in cellml.getiterator(cellml_namespace + "group"):
    children = group.getchildren()
    if children and children[0].attrib.get("relationship") == \
           "containment": #encapsulation
        encapsulations = get_encapsulation(children[1:])

#connections = comp
components = defaultdict(dict)
for comp in cellml.getiterator(cellml_namespace + "component"):
    components[comp.attrib["name"]] = dict(variables=dict())
    # Collect variables and equations
    variables = OrderedDict()
    equations = OrderedDict()
    state_variables = OrderedDict()

    # Get variable and initial values
    for var in comp.getiterator(cellml_namespace + "variable"):

        var_name = var.attrib["name"]
        #if var_name in _all_keywords:
        #    var_name = var_name + "_"

        # Store variables
        variables[var_name] = var.attrib.get("initial_value")

        # Get equations
    for math in comp.getiterator("{http://www.w3.org/1998/Math/MathML}math"):
        for eq in math.getchildren():
            equation_list, state_variable, derivative, \
                    used_variables = mathmlparser.parse(eq)

            # Get equation name
            eq_name = equation_list[0]
            
            #if eq_name in _all_keywords:
            #    equation_list[0] = eq_name + "_"
            #    eq_name = equation_list[0]
            
            # Discard collected equation name from used variables
            used_variables.discard(eq_name)
            
            assert(re.findall("(\w+)", eq_name)[0]==eq_name)
            assert(equation_list[1] == mathmlparser["eq"])
            equations[eq_name] = "".join(equation_list[2:])
            
            # Do not register state variables twice
            if state_variable is not None and \
                   state_variable not in state_variables:
                state_variables[state_variable] = derivative

    state_variables = OrderedDict((state, variables.pop(state, None))\
                                  for state in state_variables)

    parameters = OrderedDict((name, value) for name, value in \
                             variables.items() if value is not None)

    components[comp.attrib["name"]]["states"] = state_variables
    components[comp.attrib["name"]]["parameters"] = parameters
    components[comp.attrib["name"]]["equations"] = equations
    
    for var in comp.getiterator(cellml_namespace+"variable"):
        components[comp.attrib["name"]]["variables"][var.attrib["name"]] = dict(var.attrib)

        
#for con in cellml.getiterator(cellml_namespace + "connection"):
#    con_map = con.getiterator(cellml_namespace+"map_components")[0]
#    comp1 = con_map.attrib["component_1"]
#    comp2 = con_map.attrib["component_2"]
#    
#    direction = 0
#    
#    for var_map in con.getiterator(cellml_namespace+"map_variables"):
#        var1 = var_map.attrib["variable_1"]
#        var2 = var_map.attrib["variable_2"]
#        
#        if var1 != var2:
#            print "Variable name-change!!"
#
#        var1_attr = components[comp1]["variables"][var1]
#        var2_attr = components[comp2]["variables"][var2]
#        
#        var1_interface = var1_attr.get("public_interface") or \
#                         var1_attr.get("private_interface")
#        var2_interface = var2_attr.get("public_interface") or \
#                         var2_attr.get("private_interface")
#        
#        assert var1_interface != var2_interface or var1_interface == "in"

        #if not direction:
        #    direction =  1 if components[comp1]["variables"][var1]\
        #                ["public_interface"] == "out" else -1
        #else:
        #    assert direction == 1 if components[comp1]["variables"][var1]\
        #           ["public_interface"] == "out" else -1

    #if direction == 1:
    #    components[comp1]["used_in"].append(comp2)
    #    components[comp2]["dependent_on"].append(comp1)
    #else:
    #    components[comp2]["used_in"].append(comp1)
    #    components[comp1]["dependent_on"].append(comp2)
        
        
#enacapsulation_groups = groups
