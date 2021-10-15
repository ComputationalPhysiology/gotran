import matplotlib.pyplot as plt
import numpy as np
import tentusscher_panfilov_2006_M_cell as model
from scipy.integrate import odeint

# Initial states
y0 = model.init_state_values()
# Parameters
parameters = model.init_parameter_values()

# Time steps
tsteps = np.arange(0, 400, 0.1)

# Solve ODE
y = odeint(model.rhs, y0, tsteps, args=(parameters,))

# Extract the membrane potential
V_idx = model.state_indices("V")
V = y.T[V_idx]

# Extract monitored values
monitor = np.array([model.monitor(r, t, parameters) for r, t in zip(y, tsteps)])
i_Kr_idx = model.monitor_indices("i_Kr")
i_Kr = monitor.T[i_Kr_idx]


fig, ax = plt.subplots(1, 2)
ax[0].plot(tsteps, V)
ax[0].set_title("State V")

ax[1].plot(tsteps, i_Kr)
ax[1].set_title("Monitor iKr")

fig.tight_layout()
fig.savefig("results_python")
