__author__ = "Johan Hake (hake.dev@gmail.com)"
__date__ = "2013-03-13 -- 2013-03-13"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__  = "GNU LGPL Version 3.0 or later"

from gotran import *
from scipy.integrate import odeint
import pylab
import numpy as np

# Convert CellML file to a gotran ode
ode = cellml2ode("tentusscher_noble_noble_panfilov_2004_a.cellml")

# Compile executeable code from gotran ode
tentusscher = compile_module(ode, rhs_args="stp", language="python")

# Use scipy to integrate model
t0 = 0.
t1 = 600.
dt = 0.1

rhs = tentusscher.rhs
y0 = tentusscher.init_values()
params = tentusscher.default_parameters()#stim_amplitude=1)

tsteps = np.linspace(t0, t1, t1/dt+1)
results = odeint(rhs, y0, tsteps, args=(params,))

# Plot voltage
voltage = []
ind_V = tentusscher.state_indices("V")
for res in results:
    voltage.append(res[ind_V])

pylab.plot(tsteps, voltage)
pylab.show()
