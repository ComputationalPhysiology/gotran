import numpy as np
import unittest

from gotran.model.loadmodel import CellModel
from gotran import ODESolver


class TestSolver(unittest.TestCase):

    u0 = 0.5
    v0 = -1.0

    @property
    def ode(self):

        ode = CellModel("Aslak")

        eps_value = 1.0

        u = ode.add_state("u", self.u0)
        v = ode.add_state("v", self.v0)

        eps = ode.add_parameter("eps", eps_value)

        ode.add_derivative(u, ode.t, -(1.0 / eps) * v ** 3)
        ode.add_derivative(v, ode.t, (1.0 / eps) * u ** 3)

        ode.finalize()

        return ode

    def solve(self, method):

        solver = ODESolver(self.ode, method)
        t0 = 0.0
        dt = 0.001
        t1 = 1.0
        tsteps = np.linspace(t0, t1, int(t1 // dt) + 1)

        t, results = solver.solve(tsteps)

        U = results.T[0]
        V = results.T[1]

        uv = U ** 4 + V ** 4
        uv0 = self.u0 ** 4 + self.v0 ** 4

        err = np.linalg.norm(np.subtract(uv[:-1], uv0))

        self.assertTrue(err < 2e-5)


def function_closure(method):
    def test(self):
        print("Testing solver with method {}".format(method))
        self.solve(method)

    return test


for method in ["scipy", "cvode"]:
    setattr(TestSolver, "test_" + method, function_closure(method))


if __name__ == "__main__":
    unittest.main()
