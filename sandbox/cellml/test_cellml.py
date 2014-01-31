from xml.etree import ElementTree
from collections import defaultdict, OrderedDict
from gotran.input.cellml2 import CellMLParser
from gotran.model.loadmodel import load_ode
from modelparameters.utils import list_timings, clear_timings
from gotran import warning
import glob

import re

#model = "winslow_rice_jafri_marban_ororke_1999.cellml"
model = "iyer_mazhari_winslow_2004.cellml"
model = "severi_fantini_charawi_difrancesco_2012.cellml"
#model = "terkildsen_niederer_crampin_hunter_smith_2008.cellml"
model = "Pandit_Hinch_Niederer.cellml"
#model = "grandi_pasqualini_bers_2010.cellml"
#model = "maltsev_2009_paper.cellml"
model = "nash_panfilov_2004.cellml"

extract_equations = []
change_state_names = []

#params = CellMLParser.default_parameters()
#parser = CellMLParser(model, params=params)
#open(parser.name+".ode", "w").write(parser.to_gotran())
#ode = load_ode(parser.name+".ode")

for f in glob.glob("*.cellml"):
    #print
    #print f
    params = CellMLParser.default_parameters()
    parser = CellMLParser(f, params=params)
    open(parser.name+".ode", "w").write(parser.to_gotran())
    try:
        ode = load_ode(parser.name+".ode")
    except Exception, e:
        print "***Error: Could not load gotran model", parser.name, e
        print 
    list_timings()
    clear_timings()
    print 

mathmlparser = parser.mathmlparser
parsed_components = parser.components
cellml = parser.cellml

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

si_unit_map = {"ampere":"A", "becquerel":"Bq", "candela":"cd", "celsius":"gradC",
               "coulomb":"C","dimensionless":"1", "farad":"F", "gram":"g",
               "gray":"Gy", "henry":"H", "hertz":"Hz", "joule":"J", "katal":"kat",
               "kelvin":"K", "kilogram":"kg", "liter":"l", "litre":"l",
               "lumen":"lm", "lux":"lx", "meter":"m", "metre":"m", "mole":"mole",
               "newton":"N", "ohm":"Omega", "pascal":"Pa", "radian":"rad",
               "second":"s", "siemens":"S", "sievert":"Sv", "steradian":"sr",
               "tesla":"T", "volt":"V", "watt":"W", "weber":"Wb"}

prefix_map = {"deca":"da", "hecto":"h", "kilo":"k", "mega":"M", "giga":"G",
              "tera":"T", "peta":"P", "exa":"E", "zetta":"Z", "yotta":"Y",
              "deci":"d", "centi":"c", "milli":"m", "micro":"u", "nano":"n",
              "pico":"p", "femto":"f", "atto":"a", "zepto":"z", "yocto":"y",
              None:""}

collected_units = {}

for units in parser.get_iterator("units"):
    unit_name = units.attrib["name"]
    collected_parts = OrderedDict()
    for unit in units.getchildren():
        if unit.attrib.get("multiplier"):
            warning("skipping multiplier in unit {0}".format(units.attrib["name"]))
        if unit.attrib.get("multiplier"):
            warning("skipping multiplier in unit {0}".format(units.attrib["name"]))
        cellml_unit = unit.attrib.get("units")
        prefix = prefix_map[unit.attrib.get("prefix")]
        exponent = unit.attrib.get("exponent", "1")
        if cellml_unit in si_unit_map:
            abbrev = si_unit_map[cellml_unit]
            name = prefix+abbrev
            if exponent not in ["0", "1"]:
                fullname = name + "**" + exponent
            else:
                fullname = name

            collected_parts[name] = (fullname, exponent)
        elif cellml_unit in collected_units:
            if prefix:
                warning("Skipping prefix of unit '{0}'".format(cellml_unit))
            for name, (fullnam, part_exponent) in collected_units[cellml_unit].items():
                new_exponent = str(int(part_exponent) * int(exponent))
                if new_exponent not in ["0", "1"]:
                    fullname = name + "**" + new_exponent
                else:
                    fullname = name
                
                collected_parts[name] = (fullname, exponent)
            
        else:
            #warning("Unknown unit '{0}'".format(cellml_unit))
            break
        
    collected_units[unit_name] = collected_parts
        
#    print unit_name, "=", "*".join(fullname for fullname, exp in collected_parts.values())
    
def parse_component(comp):

    # Collect variables and equations
    component = dict()
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

    component["states"] = state_variables
    component["parameters"] = parameters
    component["equations"] = equations
    component["variables"] = dict()
    
    for var in comp.getiterator(cellml_namespace+"variable"):
        component["variables"][var.attrib["name"]] = dict(var.attrib)

    return component

#connections = comp
#components = defaultdict(dict)

#for model in parser.get_iterator("import"):
#    import_comp_names = dict()
#
#    for comp in parser.get_iterator("component", model):
#        import_comp_names[comp.attrib["component_ref"]] = comp.attrib["name"]
#        
#    model_cellml = ElementTree.parse(\
#        open(model.attrib["{http://www.w3.org/1999/xlink}href"])).getroot()
#
#    for comp in model_cellml.getiterator(cellml_namespace + "component"):
#        if comp.attrib["name"] in import_comp_names:
#            new_name = import_comp_names[comp.attrib["name"]]
#            components[new_name] = parse_component(comp)
#
#for comp in cellml.getiterator(cellml_namespace + "component"):
#    if comp.attrib["name"] in components:
#        continue
#    components[comp.attrib["name"]] = parse_component(comp)
#
#new_variable_names = dict()
#same_variable_names = dict()

#for con in cellml.getiterator(cellml_namespace + "connection"):
#    con_map = con.getiterator(cellml_namespace+"map_components")[0]
#    comp1 = con_map.attrib["component_1"]
#    comp2 = con_map.attrib["component_2"]
#    
#    direction = 0
#
#    #print comp1, comp2
#    
#    for var_map in con.getiterator(cellml_namespace+"map_variables"):
#        var1 = var_map.attrib["variable_1"]
#        var2 = var_map.attrib["variable_2"]
#        
#        if var1 != var2:
#            if comp1 not in new_variable_names:
#                new_variable_names[comp1] = {var1:defaultdict(list)}
#            elif var1 not in new_variable_names[comp1]:
#                new_variable_names[comp1][var1] = defaultdict(list)
#            new_variable_names[comp1][var1][var2].append(comp2)
#
#        else:
#
#            if comp1 not in same_variable_names:
#                same_variable_names[comp1] = {var1:[comp2]}
#            elif var1 not in same_variable_names[comp1]:
#                same_variable_names[comp1][var1] = [comp2]
#            else:
#                same_variable_names[comp1][var1].append(comp2)
#            
#            #print "Variable name-change!!", var1, var2
#
#        var1_attr = components[comp1]["variables"][var1]
#        var2_attr = components[comp2]["variables"][var2]
#        
#        var1_interface = var1_attr.get("public_interface") or \
#                         var1_attr.get("private_interface")
#        var2_interface = var2_attr.get("public_interface") or \
#                         var2_attr.get("private_interface")
#        
        #assert var1_interface != var2_interface or var1_interface == "in"

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
