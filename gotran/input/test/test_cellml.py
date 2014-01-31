__author__ = "Johan Hake (hake.dev@gmail.com)"
__date__ = "2012-05-07 -- 2014-01-31"
__copyright__ = "Copyright (C) 2012 " + __author__
__license__  = "GNU LGPL Version 3.0 or later"

import unittest
from gotran.input.cellml import *
from gotran.codegeneration.compilemodule import compile_module
from gotran.common.options import parameters

supported_models_form = """
\documentclass[a4paper,12pt]{{article}}
\usepackage{{fullpage}}
\usepackage{{longtable}}
\usepackage{{multicol}}
\usepackage{{amsmath}}
\usepackage{{mathpazo}}
\usepackage[mathpazo]{{flexisym}}
\usepackage{{breqn}}
\setkeys{{breqn}}{{breakdepth={{1}}}}
\\begin{{document}}
\section{{Supported and tested CellML models in Gotran}}
\\begin{{itemize}}
{0}
\end{{itemize}}
\section{{Gotran fails to convert the following CellML models}}
\\begin{{itemize}}
{1}
\end{{itemize}}
\end{{document}}
"""

cellml_data = dict(\
    terkildsen_niederer_crampin_hunter_smith_2008 = dict(\
        num_states=22),
    Pandit_Hinch_Niederer = dict(\
        num_states=22),
    Pandit_et_al_2001_endo = dict(\
        num_states=26),
    niederer_hunter_smith_2006 = dict(\
        num_states=5),
    iyer_mazhari_winslow_2004 = dict(\
        num_states=67),
    shannon_wang_puglisi_weber_bers_2004_b = dict(\
        num_states=45),
    maleckar_greenstein_trayanova_giles_2009 = dict(\
        num_states=30),
    irvine_jafri_winslow_1999 = dict(\
        num_states=13),
    grandi_pasqualini_bers_2010 = dict(\
        num_states=39),
    rice_wang_bers_detombe_2008 = dict(\
        num_states=11),
    maltsev_2009_paper = dict(\
        num_states=29),
    severi_fantini_charawi_difrancesco_2012 = dict(\
        num_states=33, change_state_names=[]),
    tentusscher_noble_noble_panfilov_2004_a = dict(\
        num_states=17),
    ten_tusscher_model_2006_IK1Ko_M_units = dict(\
        num_states=19),
    winslow_rice_jafri_marban_ororke_1999 = dict(\
        num_states=33),
)

not_supported = cellml_data.keys()
supported_models = []

skip = dict(shannon_wang_puglisi_weber_bers_2004_b = "the model exhibits unstable numerics",
            Pandit_Hinch_Niederer = "Some math trouble",
            iyer_mazhari_winslow_2004 = "Some math trouble",
            winslow_rice_jafri_marban_ororke_1999 = "Some math trouble",
            severi_fantini_charawi_difrancesco_2012 = "Some translation trouble...",
            terkildsen_niederer_crampin_hunter_smith_2008 = "Some translation trouble...",
            )

test_form = """
class {name}Tester(unittest.TestCase, TestBase):
    def setUp(self):
        if "change_state_names" in cellml_data["{name}"]:
            params = dict(change_state_names=cellml_data["{name}"]["change_state_names"])
            self.ode = cellml2ode("{name}.cellml", **params)
        else:
            self.ode = cellml2ode("{name}.cellml")
            
        self.name = "{name}"

#    def test_num_statenames(self):
#        self.assertEqual(self.ode.num_states, {num_states})

"""

# Copy of default parameters
generation = parameters.generation.copy()

# Set what code we are going to generate and not
for what_not in ["componentwise_rhs_evaluation",
                 "forward_backward_subst",
                 "linearized_rhs_evaluation",
                 "lu_factorization",
                 "monitored",
                 "jacobian"]:
    generation.functions[what_not].generate = False

# Only generate rhs
generation.functions.rhs.generate = True

class TestBase(object):
    def test_run(self):
        from scipy.integrate import odeint
        import numpy as np
        from cPickle import load

        self.assertEqual(self.ode.num_states, cellml_data[self.name]["num_states"])

        # Load reference data
        data = load(open(self.ode.name+".cpickle"))
        ref_time = data.pop("time")
        state_name = data.keys()[0]
        ref_data = data[state_name]

        # Compile ODE
        module = compile_module(self.ode, language="C", generation_params=generation)
        rhs = module.rhs
        y0 = module.init_state_values()
        model_params = module.init_parameter_values()
        
        # Run simulation
        t0 = ref_time[0]
        t1 = ref_time[-1]
        dt = min(0.1, t1/1000000.)
        tsteps = np.linspace(t0, t1, int(t1/dt)+1)
        results = odeint(rhs, y0, tsteps, args=(model_params,))
        
        # Get data to compare with ref
        ind_state = module.state_indices(state_name)
        comp_data = []
        for res in results:
            comp_data.append(res[ind_state])
        
        comp_data = np.array(comp_data)
        
        ref_interp_data = np.interp(tsteps, ref_time, ref_data)
        data_range = ref_interp_data.max()-ref_interp_data.min()
        rel_diff = np.abs((ref_interp_data-comp_data)/data_range).sum()/len(comp_data)

        if do_plot:
            import pylab
            pylab.plot(tsteps, comp_data, ref_time, ref_data)
            pylab.legend(["Gotran data", "Ref CellML"])
            pylab.title(self.name.replace("_", "\_"))
            pylab.show()
    
        print "Rel diff:", rel_diff
        self.assertTrue(rel_diff<1e-3)
        supported_models.append(not_supported.pop(not_supported.index(self.name)))

test_code = []
for name, data in cellml_data.items():
    if name in skip:
        print "Skipping:", name, "because", skip[name]
        continue
    test_code.append(test_form.format(name=name, **data))

exec("\n\n".join(test_code))
do_plot = False

if __name__ == "__main__":

    try:
        unittest.main()
    except:

        not_supported = [not_sup + "\\\\{{\\it {0}}}".format(skip.get(not_sup, "").capitalize()) for not_sup in not_supported]

        supported_str = "\n".join("\item {0}".format(\
            " ".join(part.capitalize() for part in model.split("_"))) \
                                  for model in supported_models)
        not_supported_str = "\n".join("\item {0}".format(\
            " ".join(part.capitalize() for part in model.split("_"))) \
                                  for model in not_supported)
        
        #print supported_str
        open("supported_cellml_models.tex", "w").write(supported_models_form.format(\
            supported_str, not_supported_str))
