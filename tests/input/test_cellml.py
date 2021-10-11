__author__ = "Johan Hake (hake.dev@gmail.com)"
__date__ = "2012-05-07 -- 2015-03-03"
__copyright__ = "Copyright (C) 2012 " + __author__
__license__ = "GNU LGPL Version 3.0 or later"

import unittest
import glob
import pytest
from gotran.input.cellml import cellml2ode
from gotran.codegeneration.compilemodule import compile_module
from gotran.common.options import parameters
from pathlib import Path
from scipy.integrate import odeint
import numpy as np
import pickle

_here = Path(__file__).absolute().parent

supported_models_form = """
\documentclass[a4paper,12pt]{{article}}
\\usepackage{{fullpage}}
\\usepackage{{longtable}}
\\usepackage{{multicol}}
\\usepackage{{amsmath}}
\\usepackage{{mathpazo}}
\\usepackage[mathpazo]{{flexisym}}
\\usepackage{{breqn}}
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

# not_supported = cellml_models[:]
supported_models = []

skip = dict(
    Pandit_Hinch_Niederer="Some units trouble",
    Niederer_et_al_2006="NameError: name 'J_TRPN' is not defined",
    iyer_mazhari_winslow_2004="NameError: name 'INa' is not defined",
    severi_fantini_charawi_difrancesco_2012="NameError: name 'i_f' is not defined",
    terkildsen_niederer_crampin_hunter_smith_2008="NameError: name 'I_Na' is not defined",
    niederer_hunter_smith_2006="NameError: name 'J_TRPN' is not defined",
    winslow_rice_jafri_marban_ororke_1999=(
        "self.assertTrue(rel_diff<6e-3), "
        "AssertionError: False is not true, "
        "Rel diff: 0.3952307989374001"
    ),
    maleckar_greenstein_trayanova_giles_2009=("assert 0.5240054057725506 < 0.006"),
)
cellml_models = [
    model
    for model in glob.glob(_here.joinpath("*.cellml").as_posix())
    if Path(model).stem not in skip
]


# Copy of default parameters
generation = parameters.generation.copy()

# Set what code we are going to generate and not
for what_not in [
    "componentwise_rhs_evaluation",
    "forward_backward_subst",
    "linearized_rhs_evaluation",
    "lu_factorization",
    "monitored",
]:
    generation.functions[what_not].generate = False

# Only generate rhs and jacobian
generation.functions.rhs.generate = True
generation.functions.jacobian.generate = False


# self.assertEqual(self.ode.num_states, cellml_data[self.name]["num_states"])

# Load reference data
@pytest.mark.parametrize("path", cellml_models)
def test_cellml(path):
    path = Path(path)
    ode = cellml2ode(path)

    with open(ode.name + ".cpickle", "rb") as f:

        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        data = u.load()

    ref_time = data.pop("time")
    state_name = list(data.keys())[0]
    ref_data = data[state_name]

    # Compile ODE
    module = compile_module(ode, language="Python", generation_params=generation)
    rhs = module.rhs
    jac = None  # module.compute_jacobian
    y0 = module.init_state_values()
    model_params = module.init_parameter_values()

    # Run simulation
    t0 = ref_time[0]
    t1 = ref_time[-1]
    print(t0, t1)
    dt = min(0.1, t1 / 100000.0)
    tsteps = np.linspace(t0, t1, int(t1 / dt) + 1)
    # print("using", dt, "to integrate", name)
    results = odeint(rhs, y0, tsteps, Dfun=jac, args=(model_params,))

    # Get data to compare with ref
    ind_state = module.state_indices(state_name)
    comp_data = []
    for res in results:
        comp_data.append(res[ind_state])

    comp_data = np.array(comp_data)

    ref_interp_data = np.interp(tsteps, ref_time, ref_data)
    data_range = ref_interp_data.max() - ref_interp_data.min()
    rel_diff = np.abs((ref_interp_data - comp_data) / data_range).sum() / len(comp_data)

    # if do_plot:
    #     import pylab

    #     pylab.plot(tsteps, comp_data, ref_time, ref_data)
    #     pylab.legend(["Gotran data", "Ref CellML"])
    #     pylab.title(self.name.replace("_", "\_"))
    #     pylab.show()

    print("Rel diff:", rel_diff)
    assert rel_diff < 6e-3