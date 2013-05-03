__author__ = "Johan Hake (hake.dev@gmail.com)"
__date__ = "2012-05-07 -- 2013-05-03"
__copyright__ = "Copyright (C) 2012 " + __author__
__license__  = "GNU LGPL Version 3.0 or later"

import unittest
from gotran.input.cellml import *
from gotran import compile_module

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

cellml_data = dict(
    terkildsen_niederer_crampin_hunter_smith_2008 = dict(\
        num_states=22, extract_equations=["FVRT", "FVRT_Ca", "Ca_b"], \
        change_state_names=[]),
    Pandit_Hinch_Niederer = dict(\
        num_states=6, extract_equations=[], change_state_names=[]),
    Pandit_et_al_2001_endo = dict(\
        num_states=26, extract_equations=[], change_state_names=[]),
    niederer_hunter_smith_2006 = dict(\
        num_states=5, extract_equations=["Ca_b", "Ca_i"], change_state_names=[]),
    iyer_mazhari_winslow_2004 = dict(\
        num_states=67, extract_equations=["RT_over_F"], change_state_names=[]),
    shannon_wang_puglisi_weber_bers_2004_b = dict(\
        num_states=45, extract_equations=[], change_state_names=["R"]),
    maleckar_greenstein_trayanova_giles_2009 = dict(\
        num_states=30, extract_equations=[], change_state_names=[]),
    irvine_jafri_winslow_1999 = dict(\
        num_states=13, extract_equations=[], change_state_names=[]),
    grandi_pasqualini_bers_2010 = dict(\
        num_states=39, extract_equations=[], change_state_names=[]),
    rice_wang_bers_detombe_2008 = dict(\
        num_states=11, extract_equations=[], change_state_names=[]),
    maltsev_2009_paper = dict(\
        num_states=29, extract_equations=[], change_state_names=["R"]),
    severi_fantini_charawi_difrancesco_2012 = dict(\
        num_states=33, extract_equations=["RTONF", "V", "Nai"],
        change_state_names=["R"]),
    tentusscher_noble_noble_panfilov_2004_a = dict(\
        num_states=17, extract_equations=[],
        change_state_names=[]),
    ten_tusscher_model_2006_IK1Ko_M_units = dict(\
        num_states=19, extract_equations=[], change_state_names=[]),
    winslow_rice_jafri_marban_ororke_1999 = dict(\
        num_states=33, extract_equations=[], change_state_names=[]),
    )

not_supported = cellml_data.keys()
supported_models = []

skip = dict(grandi_pasqualini_bers_2010 = "the model use time differential in assignment",
            terkildsen_niederer_crampin_hunter_smith_2008 = "the encapsulation structure of the model is not properly parsed",
            Pandit_Hinch_Niederer = "the model is not translated correct",
            # Could fix this by always assign derivatives in component
            iyer_mazhari_winslow_2004 = "the model use intermediates with same name in assignment of derivatives",
            shannon_wang_puglisi_weber_bers_2004_b = "the model exhibits unstable numerics"
            )

test_form = """
class {name}Tester(unittest.TestCase, TestBase):
    def setUp(self):
        self.ode = cellml2ode("{name}.cellml",
                              extract_equations={extract_equations},
                              change_state_names={change_state_names})
        self.name = "{name}"

    def test_num_statenames(self):
        self.assertEqual(self.ode.num_states, {num_states})

"""

class TestBase(object):
    def test_run(self):
        from scipy.integrate import odeint
        import numpy as np
        from cPickle import load

        # Load reference data
        data = load(open(self.ode.name+".cpickle"))
        ref_time = data.pop("time")
        state_name = data.keys()[0]
        ref_data = data[state_name]

        # Compile ODE
        module = compile_module(self.ode, rhs_args="stp", language="C")
        rhs = module.rhs
        y0 = module.init_values()
        model_params = module.default_parameters()
        
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
        
        print supported_str
        open("supported_cellml_models.tex", "w").write(supported_models_form.format(\
            supported_str, not_supported_str))
