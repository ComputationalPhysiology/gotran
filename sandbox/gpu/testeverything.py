import itertools as it
import traceback
from io import StringIO

import cudaodesystemsolver as coss
import numpy as np
from goss import Progress

from gotran import load_ode
from gotran.common import clear_timings
from gotran.common import list_timings


def get_dtype_str(float_precision):
    return {False: "float32", True: "float64"}[float_precision]


def get_float_prec_str(float_precision):
    return {False: "single", True: "double"}[float_precision]


stored_field_states = list()


def get_store_field_states_fn(num_nodes):
    i = len(stored_field_states)
    stored_field_states.append(list())

    def store_field_states_fn(field_states):
        stored_field_states[i].append(
            [field_states[0], field_states[num_nodes / 2], field_states[num_nodes - 1]],
        )

    return store_field_states_fn


def get_store_all_field_states_fn(num_nodes):
    i = len(stored_field_states)
    stored_field_states.append(list())

    def store_field_states_fn(field_states):
        stored_field_states[i].append(field_states.copy())

    return store_field_states_fn


def get_g_to_field_parameter_values(num_nodes, float_precision):
    return (
        0.294
        * np.arange(0, num_nodes, dtype=get_dtype_str(float_precision))[::-1]
        / (num_nodes - 1.0)
    )


tentusscher_fname = "tentusscher_panfilov_2006_M_cell.ode"


class TestCase(object):
    def __init__(
        self,
        ode,
        num_nodes,
        dt,
        tstop,
        t0=0.0,
        solver="rush_larsen",
        field_states=[""],
        field_states_getter_fn=None,
        field_parameters=[""],
        field_parameter_values_getter_fn=None,
        block_size=1024,
        double=True,
        statesrepr="named",
        paramrepr="named",
        bodyrepr="named",
        use_cse=False,
        update_host_states=False,
        update_field_states=True,
    ):
        self.ode = load_ode(ode)
        self.num_nodes = num_nodes
        self.dt = dt
        self.tstop = tstop
        self.t0 = t0
        self.solver = solver
        self.field_states = field_states
        self.field_states_fn = (
            field_states_getter_fn(num_nodes)
            if field_states_getter_fn is not None
            else None
        )
        self.field_parameters = field_parameters
        self.field_parameter_values = (
            field_parameter_values_getter_fn(num_nodes, double)
            if field_parameter_values_getter_fn is not None
            else None
        )
        self.block_size = block_size
        self.double = double
        self.statesrepr = statesrepr
        self.paramrepr = paramrepr
        self.bodyrepr = bodyrepr
        self.use_cse = use_cse
        self.update_host_states = update_host_states
        self.update_field_states = update_field_states


def createTestCases(
    ode,
    num_nodes,
    dt,
    tstop,
    t0=(0.0,),
    solver=("rush_larsen",),
    field_states=([""],),
    field_states_getter_fn=(None,),
    field_parameters=([""],),
    field_parameter_values_getter_fn=(None,),
    block_size=(256,),
    double=(True,),
    statesrepr=("named",),
    paramrepr=("named",),
    bodyrepr=("named",),
    use_cse=(False,),
    update_host_states=(False,),
    update_field_states=(True,),
):
    testcases = list()
    for comb in it.product(
        ode,
        num_nodes,
        dt,
        tstop,
        t0,
        solver,
        field_states,
        field_states_getter_fn,
        field_parameters,
        field_parameter_values_getter_fn,
        block_size,
        double,
        statesrepr,
        paramrepr,
        bodyrepr,
        use_cse,
        update_host_states,
        update_field_states,
    ):
        testcases.append(TestCase(*comb))

    print("Generated {0} test cases.".format(len(testcases)))
    return testcases


def testFloatPrecision(
    num_nodes=1024,
    dt=0.1,
    tstop=300.0,
    solver="rush_larsen",
    field_states=["V"],
    field_states_getter_fn=get_store_field_states_fn,
    field_parameters=["g_to"],
    field_parameter_values_getter_fn=get_g_to_field_parameter_values,
):
    testcases = createTestCases(
        (tentusscher_fname,),
        (num_nodes,),
        (dt,),
        (tstop,),
        solver=(solver,),
        field_states=(field_states,),
        field_states_getter_fn=(field_states_getter_fn,),
        field_parameters=(field_parameters,),
        field_parameter_values_getter_fn=(field_parameter_values_getter_fn,),
        double=(True,),  # , False)
    )

    print("Running FLOAT PRECISION tests...")

    names = ("double precision", "single precision")

    all_field_states, runtimes, errors = list(zip(*runTests(testcases)))

    printResults(names, all_field_states, runtimes)

    return (names, all_field_states, runtimes, testcases, errors)


def testThreadsPerBlock(
    num_nodes=1024 * 16,
    block_size=(16, 32, 64, 128, 256),
    dt=0.1,
    tstop=300.0,
    solver="rush_larsen",
    field_states=["V"],
    field_states_getter_fn=get_store_field_states_fn,
    field_parameters=["g_to"],
    field_parameter_values_getter_fn=get_g_to_field_parameter_values,
):
    testcases = createTestCases(
        (tentusscher_fname,),
        (num_nodes,),
        (dt,),
        (tstop,),
        solver=(solver,),
        field_states=(field_states,),
        field_states_getter_fn=(field_states_getter_fn,),
        field_parameters=(field_parameters,),
        field_parameter_values_getter_fn=(field_parameter_values_getter_fn,),
        block_size=block_size,
    )

    print("Running THREADS PER BLOCK tests...")

    names = ["block size {0}".format(bs) for bs in block_size]

    all_field_states, runtimes, errors = list(zip(*runTests(testcases)))

    printResults(names, all_field_states, runtimes)

    return (names, all_field_states, runtimes, testcases, errors)


def testNumNodes(
    num_nodes=[1024 * 2**n for n in range(3, 8)] + [1000 * 2**n for n in range(3, 8)],
    dt=0.1,
    tstop=300.0,
    solver="rush_larsen",
    field_states=["V"],
    field_states_getter_fn=get_store_field_states_fn,
    field_parameters=["g_to"],
    field_parameter_values_getter_fn=get_g_to_field_parameter_values,
):
    testcases = createTestCases(
        (tentusscher_fname,),
        num_nodes,
        (dt,),
        (tstop,),
        solver=(solver,),
        field_states=(field_states,),
        field_states_getter_fn=(field_states_getter_fn,),
        field_parameters=(field_parameters,),
        field_parameter_values_getter_fn=(field_parameter_values_getter_fn,),
    )

    print("Running NUM NODES tests...")

    names = ["{0} nodes".format(nn) for nn in num_nodes]

    all_field_states, runtimes, errors = list(zip(*runTests(testcases)))

    printResults(names, all_field_states, runtimes)

    return (names, all_field_states, runtimes, testcases, errors)


def testSolvers(
    num_nodes=1024,
    dt=0.001,
    tstop=50.0,
    solver=("explicit_euler", "rush_larsen", "simplified_implicit_euler"),
    field_states=["V"],
    field_states_getter_fn=get_store_field_states_fn,
    field_parameters=["g_to"],
    field_parameter_values_getter_fn=get_g_to_field_parameter_values,
):
    testcases = createTestCases(
        (tentusscher_fname,),
        (num_nodes,),
        (dt,),
        (tstop,),
        solver=solver,
        field_states=(field_states,),
        field_states_getter_fn=(field_states_getter_fn,),
        field_parameters=(field_parameters,),
        field_parameter_values_getter_fn=(field_parameter_values_getter_fn,),
    )

    print("Running SOLVERS tests...")

    names = solver

    all_field_states, runtimes, errors = list(zip(*runTests(testcases)))

    printResults(names, all_field_states, runtimes)

    return (names, all_field_states, runtimes, testcases, errors)


# TODO: Test error against dt=0.00005
def testDt(
    num_nodes=1024 * 8,
    dt=(1, 0.5, 0.2, 0.1, 0.05),
    tstop=300.0,
    solver="rush_larsen",
    field_states=["V"],
    field_states_getter_fn=get_store_field_states_fn,
    field_parameters=["g_to"],
    field_parameter_values_getter_fn=get_g_to_field_parameter_values,
):
    testcases = createTestCases(
        (tentusscher_fname,),
        (num_nodes,),
        dt,
        (tstop,),
        solver=(solver,),
        field_states=(field_states,),
        field_states_getter_fn=(field_states_getter_fn,),
        field_parameters=(field_parameters,),
        field_parameter_values_getter_fn=(field_parameter_values_getter_fn,),
    )

    print("Running DT tests...")

    names = ["dt={0}".format(dt_) for dt_ in dt]

    all_field_states, runtimes, errors = list(zip(*runTests(testcases)))

    printResults(names, all_field_states, runtimes)

    return (names, all_field_states, runtimes, testcases, errors)


def testUpdateStates(
    num_nodes=1024,
    dt=0.1,
    tstop=300.0,
    solver="rush_larsen",
    update_host_states=(True, False),
    update_field_states=(True, False),
    field_states=["V"],
    field_states_getter_fn=get_store_field_states_fn,
    field_parameters=["g_to"],
    field_parameter_values_getter_fn=get_g_to_field_parameter_values,
):
    testcases = createTestCases(
        (tentusscher_fname,),
        (num_nodes,),
        (dt,),
        (tstop,),
        solver=(solver,),
        field_states=(field_states,),
        field_states_getter_fn=(None,),
        field_parameters=(field_parameters,),
        field_parameter_values_getter_fn=(field_parameter_values_getter_fn,),
        update_host_states=update_host_states,
        update_field_states=update_field_states,
    )

    print("Running UPDATE HOST/FIELD STATES tests...")

    names = [
        "host={0}, field={1}".format(h, f)
        for h, f in it.product(update_host_states, update_field_states)
    ]

    all_field_states, runtimes, errors = list(zip(*runTests(testcases)))

    printResults(names, all_field_states, runtimes)

    return (names, all_field_states, runtimes, testcases, errors)


def testRepresentation(
    num_nodes=1024 * 16,
    dt=0.1,
    tstop=300.0,
    solver="rush_larsen",
    statesrepr=("named",),  # 'array'),
    paramrepr=("named",),  # 'array', 'numerals'),
    bodyrepr=("named",),  # 'array', 'reused_array'),
    use_cse=(False, True),  # FIXME
    field_states=["V"],
    field_states_getter_fn=get_store_field_states_fn,
    field_parameters=["g_to"],
    field_parameter_values_getter_fn=get_g_to_field_parameter_values,
):
    # FIXME: This is broken
    testcases = createTestCases(
        (tentusscher_fname,),
        (num_nodes,),
        (dt,),
        (tstop,),
        solver=(solver,),
        field_states=(field_states,),
        field_states_getter_fn=(field_states_getter_fn,),
        field_parameters=(field_parameters,),
        field_parameter_values_getter_fn=(field_parameter_values_getter_fn,),
        statesrepr=statesrepr,
        paramrepr=paramrepr,
        bodyrepr=bodyrepr,
        use_cse=use_cse,
    )

    print("Running REPRESENTATION/CSE tests...")

    names = [
        "s={0}, p={1}, b={2}, cse={3}".format(s, p, b, cse)
        for s, p, b, cse in it.product(statesrepr, paramrepr, bodyrepr, use_cse)
    ]

    all_field_states, runtimes, errors = list(zip(*runTests(testcases)))

    printResults(names, all_field_states, runtimes)

    return (names, all_field_states, runtimes, testcases, errors)


def testEverything():
    # TODO: Add testRepresentation when it's not broken

    results = dict()

    tests = (
        ("FLOAT PRECISION", testFloatPrecision),
        ("THREADS PER BLOCK", testThreadsPerBlock),
        ("NUM NODES", testNumNodes),
        ("SOLVERS", testSolvers),
        ("DT", testDt),
        ("UPDATE HOST/FIELD STATES", testUpdateStates),
        ("REPRESENTATION", testRepresentation),
    )

    for name, test in tests:
        results[name] = test()

    for name, result in results.items():
        print("\nResults from {0} test:".format(name))
        subnames, fstates, runtimes, _, _ = result
        printResults(subnames, fstates, runtimes, indent=4)

    return results


def printResults(names, all_field_states, runtimes, indent=0):
    for name, fstates, runtime in zip(names, all_field_states, runtimes):
        print(
            " " * indent
            + "{0:{1}s}: {2:{3}.2f}s ({4}, {5}, {6})".format(
                name,
                max(list(map(len, names))),
                runtime,
                max(list(map(len, list(map(str, list(map(int, runtimes))))))) + 3,
                fstates[0],
                fstates[len(fstates) / 2],
                fstates[-1],
            ),
        )


def runTests(testcases, printTimings=True):
    results = list()

    ntests = len(testcases)

    for i, testcase in enumerate(testcases):
        params = coss.CUDAODESystemSolver.default_parameters()
        params.solver = testcase.solver
        params.code.states.field_states = testcase.field_states
        params.code.parameters.field_parameters = testcase.field_parameters
        params.block_size = testcase.block_size
        params.code.float_precision = get_float_prec_str(testcase.double)
        params.code.states.representation = testcase.statesrepr
        params.code.parameters.representation = testcase.paramrepr
        params.code.body.representation = testcase.bodyrepr
        params.code.body.use_cse = testcase.use_cse

        try:
            solver = coss.CUDAODESystemSolver(
                testcase.num_nodes,
                testcase.ode,
                init_field_parameters=testcase.field_parameter_values,
                params=params,
            )

            kernel_fname = solver._dump_kernel_code()
            print("Dumped CUDA code into '{0}.'".format(kernel_fname))

            solver.simulate(
                testcase.t0,
                testcase.dt,
                testcase.tstop,
                field_states_fn=testcase.field_states_fn,
                update_field_states=testcase.update_field_states,
                update_host_states=testcase.update_host_states,
            )

            if printTimings:
                list_timings()
                clear_timings()

            solver.get_field_states()

            results.append([solver.field_states, solver.simulation_runtime, None])

            solver.reset()

            print(
                "Completed test {0}/{1} in {2:.2f}s.".format(
                    i + 1,
                    ntests,
                    results[-1][1],
                ),
            )
        except Exception:
            f = StringIO()
            traceback.print_exc(file=f)
            f.read()
            results.append(["ERROR", 0.0, f.buf])
            f.close()
            print("FAILED test {0}/{1}.".format(i + 1, ntests))

    return results


def runTestsStep(
    testcases,
    printTimings=True,
    checkNaN=False,
    update_host_states=False,
):
    results = list()

    ntests = len(testcases)

    for i, testcase in enumerate(testcases):
        params = coss.CUDAODESystemSolver.default_parameters()
        params.solver = testcase.solver
        params.code.states.field_states = testcase.field_states
        params.code.parameters.field_parameters = testcase.field_parameters
        params.block_size = testcase.block_size
        params.code.float_precision = get_float_prec_str(testcase.double)
        params.code.states.representation = testcase.statesrepr
        params.code.parameters.representation = testcase.paramrepr
        params.code.body.representation = testcase.bodyrepr
        params.code.body.use_cse = testcase.use_cse

        try:
            solver = coss.CUDAODESystemSolver(
                testcase.num_nodes,
                testcase.ode,
                init_field_parameters=testcase.field_parameter_values,
                params=params,
            )

            field_states = np.zeros(
                testcase.num_nodes * len(testcase.field_states),
                dtype=get_dtype_str(testcase.double),
            )

            solver.get_field_states(field_states)
            testcase.field_states_fn(field_states)

            t = testcase.t0
            dt = testcase.dt
            tstop = testcase.tstop
            p = Progress("Test {0}".format(i + 1), int(tstop / dt))
            num_nans = 0

            while t < tstop + 1e-6:
                solver.set_field_states(field_states)
                solver.forward(
                    t,
                    dt,
                    update_simulation_runtimes=True,
                    update_host_states=update_host_states,
                )
                solver.get_field_states(field_states)
                testcase.field_states_fn(field_states)

                if checkNaN:
                    n = np.isnan(field_states).sum()
                    if n > num_nans:
                        print(t, n)
                    num_nans = n

                t += dt
                p += 1

            results.append([field_states, solver.simulation_runtime, None])
            solver.reset()
            print(
                "Completed test {0}/{1} in {2:.2f}s.".format(
                    i + 1,
                    ntests,
                    results[-1][1],
                ),
            )
        except Exception:
            f = StringIO()
            traceback.print_exc(file=f)
            f.read()
            results.append(["ERROR", 0.0, f.buf])
            f.close()
            print("FAILED test {0}/{1}.".format(i + 1, ntests))

    return results
