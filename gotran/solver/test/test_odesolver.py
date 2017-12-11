from gotran.model.loadmodel import load_cell, get_model_as_python_module, load_ode, CellModel
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat


from gotran import ODESolver
import time


def test_aslak(eps_value = 1e-1, method = "cvode"):

    ode = CellModel("Aslak")

    u0 = 0.5
    v0 = -1.0
    
    u = ode.add_state("u", u0)
    v = ode.add_state("v", v0)

    eps = ode.add_parameter("eps", eps_value)

    du_dt = ode.add_derivative(u, ode.t, - (1.0/eps) * v**3)
    dv_dt = ode.add_derivative(v, ode.t, (1.0/eps) * u**3)
    
    ode.finalize()

    if method == "scipy":
        solver = ODESolver(ode, method, hmax=2e-3, h0=1e-2)
    elif method in ["cvode", "lsodar"]:
        solver = ODESolver(ode, method, maxh=2e-3, inith=1e-2, verbosity=100)

    t0 = 0.0
    dt = 0.001
    t1 = 1.0
    tsteps = np.linspace(t0, t1, t1/dt+1)

    t0 = time.clock()
    if method == "scipy":
        t, results = solver.solve(tsteps)
    elif method in ["cvode", "lsodar"]:
        t, results = solver.solve(t1, ncp_list=tsteps)
        
    tend = time.clock()
    print("Elapsed time: {}".format(tend-t0))


    U = results.T[0]
    V = results.T[1]

    uv = U**4 + V**4
    uv0 = u0**4 + v0**4

    err = np.abs(uv[1:] - uv0)
    return t[1:], err


def run_aslak():

    print("Test aslak")
    for method in ["scipy", "lsodar", "cvode"]:
        print("\nMethod:{}".format(method))
        fig, ax = plt.subplots()
        for r in range(0,5):
            print("Epsilon=1e-{}".format(r))
            ts, err = test_aslak(10**-r, method)
            ax.semilogy(ts, err, label = r"$\varepsilon = 10^{{{}}}$".format(-r))
   
        ax.legend(loc="best")
        ax.set_ylabel("$u(t)^4 + v(t)^4 - (u(0)^4 + v(0)^4)$")
        ax.set_xlabel("$t$")
        ax.set_title("Python")

        fig.savefig("ode_solver_{}.png".format(method))


def test_paci(method="scipy"):


    ode = load_cell("paci.ode")

    if method == "scipy":
        solver = ODESolver(ode, method, hmax=2e-3, h0=1e-2)
    elif method in ["cvode", "lsodar"]:
        solver = ODESolver(ode, method, maxh=2e-3, inith=1e-2, verbosity=100)

    
    t0 = 0.0
    dt = 0.01
    t1 = 10.0
    tsteps = np.linspace(t0, t1, t1/dt+1)

    t0 = time.clock()
    if method == "scipy":
        t, results = solver.solve(tsteps)
    elif method in ["cvode", "lsodar"]:
        t, results = solver.solve(t1, ncp_list=tsteps)
    t1 = time.clock()
    print("Elapsed time: {}".format(t1-t0))

    # monitor = solver.monitor(t, results)

    # fig, ax = plt.subplots(1,2)
    # ax[0].plot(t, monitor.T[solver.monitor_indices('dVm_dt')])
    # ax[0].set_title('dV_dt')

    
    # ax[1].plot(t,results.T[solver.state_indices("Vm")])
    # ax[1].set_title('V(t)')

    # fig.tight_layout()
    # plt.show()

    V =  results.T[solver.state_indices("Vm")] * 1000.0
    t = np.multiply(t, 1000.0)

    return t[-300:],V[-300:]
    
def run_paci():

    t_cvode, V_cvode = test_paci("cvode")
    t_lsodar, V_lsodar = test_paci("lsodar")
    t_scipy, V_scipy = test_paci("scipy")

    fig, ax = plt.subplots()
    ax.plot(t_cvode, V_cvode, label="cvode")
    ax.plot(t_lsodar, V_lsodar, label="lsodar")
    ax.plot(t_scipy, V_scipy, label="scipy")
    ax.legend(loc="best")
    ax.set_ylabel("V (mV)")
    ax.set_xlabel("t (ms)")
    fig.savefig("compare_paci")


    fig, ax = plt.subplots()
    ax.semilogy(t_cvode, abs(V_cvode-V_scipy), label="|cvode-scipy|")
    ax.semilogy(t_lsodar, abs(V_lsodar-V_scipy), label="|lsodar-scipy|")
    ax.semilogy(t_scipy, abs(V_cvode-V_lsodar), label="|cvode-lsodar|")
    ax.legend(loc="best")
    ax.set_ylabel("V (mV)")
    ax.set_xlabel("t (ms)")
    fig.savefig("compare_paci_diff")



if __name__ == "__main__":
    run_paci()
    run_aslak()
