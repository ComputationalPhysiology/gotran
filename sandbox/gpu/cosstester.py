from gotran import load_ode
from gotran.common import list_timings, clear_timings
from StringIO import StringIO

import ast
import cudaodesystemsolver as coss
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

try:
    import progressbar
except ImportError:
    progressbar = None
import random
import threading
import time
import traceback

DEFAULT_ODE_MODEL = "tentusscher"

ODE_MODELS = {
    "tentusscher": "tentusscher_panfilov_2006_M_cell.ode",
    "grandi": "grandi_pasqualini_bers_2010.ode",
    "beeler": "beeler_reuter_1977.ode",
}

FIELD_STATES = {"tentusscher": ["V"], "grandi": ["V_m"], "beeler": ["V"]}

FIELD_PARAMETERS = {"tentusscher": ["g_to"], "grandi": [""], "beeler": [""]}

PLOT_STRINGS = {
    "block_size": "Threads per block",
    "double": "Float precision",
    "dt": "Time step",
    "field_state_values": "Field state value",
    "field_parameter_values": "Field parameter value",
    "num_nodes": "Number of nodes",
    "runtime": "Time (s)",
    "time": "Time (s)",
}

PLOT_TITLES = {
    "double": "Float precision for {0}",
    "num_nodes": "Number of nodes for {0}",
    "runtime": "Simulation time for {0}",
}


class COSSTestCase(object):
    def __init__(self, **kwargs):
        COSSTestCase.check_kwargs(**kwargs)
        params = COSSTestCase.default_parameters()
        params.update(kwargs)

        if "field_states" in params and isinstance(params["field_states"], dict):
            params["field_states"] = params["field_states"][params["ode_model"]]

        if "field_parameters" in params and isinstance(
            params["field_parameters"], dict
        ):
            params["field_parameters"] = params["field_parameters"][params["ode_model"]]

        self.__dict__.update(params)

        print self.ode_model
        self.ode = load_ode(self.ode_model)
        self.stored_field_states = list()
        if self.field_states_getter_fn is None:
            self.stored_field_states = None
            self.field_states_fn = None
        else:
            self.stored_field_states = list()
            self.field_states_fn = self.field_states_getter_fn(
                self.num_nodes, self.stored_field_states
            )
        initial_field_params = [
            param.init
            for param in self.ode.parameters
            if param.name in self.field_parameters
        ]
        if self.field_parameter_values_getter_fn is None:
            self.field_parameter_values = None
        else:
            self.field_parameter_values = self.field_parameter_values_getter_fn(
                initial_field_params, self.num_nodes, self.double
            )

    @staticmethod
    def default_parameters():
        return {
            "ode_model": None,
            "num_nodes": None,
            "dt": None,
            "tstop": None,
            "t0": 0.0,
            "solver": "rush_larsen",
            "field_states": [""],
            "field_states_getter_fn": None,
            "field_parameters": [""],
            "field_parameter_values_getter_fn": None,
            "block_size": 256,
            "double": True,
            "statesrepr": "named",
            "paramrepr": "named",
            "bodyrepr": "named",
            "use_cse": False,
            "update_host_states": False,
            "update_field_states": True,
            "ode_substeps": 1,
            "gpu_arch": None,
            "gpu_code": None,
            "cuda_cache_dir": None,
            "nvcc_options": [""],
        }

    @staticmethod
    def required_parameters():
        return ("ode_model", "num_nodes", "dt", "tstop")

    @staticmethod
    def parameter_iterability_map():
        return {k: v == [""] for k, v in COSSTestCase.default_parameters().items()}

    @staticmethod
    def check_kwargs(**kwargs):
        for key in COSSTestCase.required_parameters():
            if key not in kwargs:
                raise ValueError("Missing required argument '{0}'".format(key))

        params = COSSTestCase.default_parameters()

        for key, value in kwargs.iteritems():
            if key not in params:
                raise ValueError("Unknown argument '{0}'".format(key))


def dict_product(dicts):
    return (dict(it.izip(dicts, prod)) for prod in it.product(*dicts.itervalues()))


def createTestCases(**kwargs):
    params = COSSTestCase.default_parameters()
    params.update(kwargs)

    param_iterability_map = COSSTestCase.parameter_iterability_map()
    named_keys = list()

    for key, value in params.iteritems():
        if (
            param_iterability_map[key]
            and isinstance(value, (list, tuple))
            and not isinstance(value[0], (list, tuple))
            or not isinstance(value, (list, tuple))
        ):
            params[key] = (value,)
        elif len(value) > 1:
            named_keys.append(key)

    testcases, names = list(), list()
    for comb in dict_product(params):
        testcases.append(COSSTestCase(**comb))
        names.append(",".join("%s=%s" % (k, comb[k]) for k in named_keys))

    print "Generated {0} test cases.".format(len(testcases))
    return (testcases, names)


def runTestSuite(
    title,
    keep_cuda_code=False,
    printTimings=True,
    saveDirectory=None,
    showProgress=True,
    **kwargs
):
    testcases, names = createTestCases(**kwargs)

    print "Running '{0}' tests...".format(title)

    results = runTests(
        testcases,
        names=names,
        keep_cuda_code=keep_cuda_code,
        printTimings=printTimings,
        showProgress=showProgress,
    )

    printResults(results)

    if saveDirectory is not None:
        saveResultsSimple(title, results, saveDirectory)

    return results


def runTestSuites(
    test_suites=None,
    keep_cuda_code=False,
    saveDirectory=None,
    printTimings=True,
    showProgress=True,
):
    test_suites = test_suites or getSampleTestSuites()

    results = dict()

    for title, test_suite in test_suites.iteritems():
        results[title] = runTestSuite(
            title,
            keep_cuda_code=keep_cuda_code,
            printTimings=printTimings,
            saveDirectory=saveDirectory,
            # clear_stored_fstates=False,
            showProgress=showProgress,
            **test_suite
        )

    for title, result in results.iteritems():
        print "Results from '{0}' test:".format(title)
        printResults(result, indent=4)

    return results


def printResults(results, indent=0):
    max_name_len = max(1, *map(len, [r["name"] for r in results]))
    max_runtime_str_len = (
        max(map(len, map(str, map(int, [r["runtime"] for r in results])))) + 3
    )
    for result in results:
        name, runtime, fstates, error = (
            result["name"],
            result["runtime"],
            result["field_states"],
            result["error"],
        )
        fstates_fmt = (
            "{0}, {1}, {2}".format(fstates[0], fstates[len(fstates) / 2], fstates[-1])
            if fstates is not None
            else "No field states"
        )
        error_fmt = "(ERROR)" if error is not None else ""
        print " " * indent + "{0:{1}s}: {2:{3}.2f}s ({4}) {5}".format(
            name, max_name_len, runtime, max_runtime_str_len, fstates_fmt, error_fmt
        )
    print


def runTestCase(
    testcase,
    name,
    keep_cuda_code=False,
    printTimings=True,
    showProgress=True,
    solver=None,
):
    params = getTestCaseParams(testcase, keep_cuda_code=keep_cuda_code)

    result = {
        "testcase": testcase,
        "name": name,
        "field_states": None,
        "runtime": 0.0,
        "error": None,
    }

    try:
        if solver is None:
            solver = coss.CUDAODESystemSolver(
                testcase.num_nodes,
                testcase.ode,
                init_field_parameters=testcase.field_parameter_values,
                params=params,
            )

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

        result["field_states"] = solver.field_states
        result["runtime"] = solver.simulation_runtime

        solver.reset()

        print "Completed test '{0}' in {1:.2f}s.".format(name, result["runtime"])
    except:
        f = StringIO()
        traceback.print_exc(file=f)
        f.read()
        result["error"] = f.buf
        f.close()
        print "FAILED test '{0}'.".format(name)

    print
    return result


def runSteppedTestCase(
    testcase,
    name,
    keep_cuda_code=False,
    printTimings=True,
    showProgress=True,
    solver=None,
):
    params = getTestCaseParams(testcase, keep_cuda_code=keep_cuda_code)

    epsilon = 1e-6
    bar = None
    result = {
        "testcase": testcase,
        "name": name,
        "field_states": None,
        "runtime": 0.0,
        "error": None,
    }

    try:
        if solver is None:
            solver = coss.CUDAODESystemSolver(
                testcase.num_nodes,
                testcase.ode,
                init_field_parameters=testcase.field_parameter_values,
                params=params,
            )

        field_states_fn = testcase.field_states_fn
        if field_states_fn is None:
            field_states_fn = lambda _: None

        if testcase.field_states is not None and testcase.field_states != [""]:
            field_states = np.zeros(
                testcase.num_nodes * len(testcase.field_states),
                dtype=getFloatPrecisionDtypeStr(testcase.double),
            )
            solver.get_field_states(field_states)

            field_states_fn(field_states)
        else:
            field_states = None

            field_states_fn(solver.field_states)

        t = testcase.t0
        dt = testcase.dt * testcase.ode_substeps
        tstop = testcase.tstop

        if progressbar is not None and showProgress:
            bar = progressbar.ProgressBar(
                maxval=tstop + epsilon,
                widgets=[
                    progressbar.Bar("=", "[", "]"),
                    " ",
                    progressbar.Percentage(),
                    " ",
                    progressbar.ETA(),
                ],
            )

        while t < tstop + epsilon:
            if testcase.update_field_states:
                solver.set_field_states(field_states)
            solver.forward(
                t,
                dt,
                update_simulation_runtimes=True,
                update_host_states=testcase.update_host_states,
            )
            if testcase.update_field_states:
                solver.get_field_states(field_states)
                if field_states is not None:
                    field_states_fn(field_states)
                else:
                    field_states_fn(solver.field_states)

            if progressbar is not None and showProgress:
                bar.update(t)

            t += dt

        if progressbar is not None and showProgress:
            bar.finish()

        if printTimings:
            list_timings()
            clear_timings()

        solver.get_field_states(field_states)

        result["field_states"] = field_states
        result["runtime"] = solver.simulation_runtime

        solver.reset()

        print "Completed test '{0}' in {1:.2f}s".format(name, result["runtime"])
    except:
        if bar is not None:
            bar.finish()
        f = StringIO()
        traceback.print_exc(file=f)
        f.read()
        result["error"] = f.buf
        f.close()
        print "FAILED test '{0}'.".format(name)

    print
    return result


def runTests(
    testcases, names=None, keep_cuda_code=False, printTimings=True, showProgress=True
):
    results = list()

    ntests = len(testcases)

    names = names or [""] * ntests

    for i, testcase, name in it.izip(xrange(ntests), testcases, names):
        print "Running test {0}/{1} ({2})...".format(i + 1, ntests, name)
        # results.append(runTestCase(testcase, name,
        #                           keep_cuda_code=keep_cuda_code,
        #                           printTimings=printTimings))
        results.append(
            runSteppedTestCase(
                testcase,
                name,
                keep_cuda_code=keep_cuda_code,
                printTimings=printTimings,
                showProgress=showProgress,
            )
        )

    return results


def getTestCaseParams(testcase, keep_cuda_code=False):
    params = coss.CUDAODESystemSolver.default_parameters()
    params.solver = testcase.solver
    params.code.states.field_states = testcase.field_states
    params.code.parameters.field_parameters = testcase.field_parameters
    params.block_size = testcase.block_size
    params.code.float_precision = getFloatPrecisionStr(testcase.double)
    params.code.states.representation = testcase.statesrepr
    params.code.parameters.representation = testcase.paramrepr
    params.code.body.representation = testcase.bodyrepr
    params.code.body.use_cse = testcase.use_cse
    params.ode_substeps = testcase.ode_substeps
    params.gpu_arch = testcase.gpu_arch
    params.gpu_code = testcase.gpu_code
    params.keep_cuda_code = keep_cuda_code
    params.cuda_cache_dir = testcase.cuda_cache_dir
    params.nvcc_options = testcase.nvcc_options
    return params


# This is really ugly, but it works
def get_store_partial_field_state_fn(num_nodes, stored_field_states, n=3):
    def store_field_states_fn(field_states):
        stored_field_states.append(
            [field_states[j * (len(field_states) - 1) / (n - 1)] for j in xrange(n)]
        )

    return store_field_states_fn


def get_c_s_partial_field_state_fn(num_nodes, stored_field_states, n=3):
    def c_s_field_states_fn(field_states):
        # Waste some time:
        for fs in field_states:
            fs * random.uniform(0.1, 10.0)

        stored_field_states.append(
            [field_states[j * (len(field_states) - 1) / (n - 1)] for j in xrange(n)]
        )

    return c_s_field_states_fn


def get_store_all_field_states_fn(num_nodes, stored_field_states):
    def store_field_states_fn(field_states):
        stored_field_states.append(field_states.copy())

    return store_field_states_fn


def linear_field_parameter_transform(field_parameters, num_nodes, float_precision):
    if len(field_parameters) == 0:
        return None
    else:
        return (
            np.concatenate(
                tuple(
                    np.reshape(
                        np.linspace(value, value * 0.01, num_nodes), (num_nodes, 1)
                    )
                    for value in field_parameters
                ),
                axis=1,
            )
            .astype(getFloatPrecisionDtypeStr(float_precision))
            .copy()
        )


def getFloatPrecisionDtypeStr(is_double):
    return "float64" if is_double else "float32"


def getFloatPrecisionStr(is_double):
    return "double" if is_double else "single"


sample_test_suites = {
    "FLOAT PRECISION": {
        "ode_model": ODE_MODELS[DEFAULT_ODE_MODEL],
        "num_nodes": 8192,
        "dt": 0.1,
        "tstop": 300.0,
        "field_states": FIELD_STATES[DEFAULT_ODE_MODEL],
        "field_states_getter_fn": get_store_partial_field_state_fn,
        "field_parameters": FIELD_PARAMETERS[DEFAULT_ODE_MODEL],
        "field_parameter_values_getter_fn": linear_field_parameter_transform,
        "double": (False, True),
    },
    "THREADS PER BLOCK": {
        "ode_model": ODE_MODELS[DEFAULT_ODE_MODEL],
        "num_nodes": 8192,
        "dt": 0.1,
        "tstop": 300.0,
        "field_states": FIELD_STATES[DEFAULT_ODE_MODEL],
        "field_states_getter_fn": get_store_partial_field_state_fn,
        "field_parameters": FIELD_PARAMETERS[DEFAULT_ODE_MODEL],
        "field_parameter_values_getter_fn": linear_field_parameter_transform,
        "block_size": (16, 32, 64, 128, 256),
    },
    "NUM NODES": {
        "ode_model": ODE_MODELS[DEFAULT_ODE_MODEL],
        "num_nodes": (1024, 2048, 4096, 8192, 16384),
        "dt": 0.1,
        "tstop": 300.0,
        "field_states": FIELD_STATES[DEFAULT_ODE_MODEL],
        "field_states_getter_fn": get_store_partial_field_state_fn,
        "field_parameters": FIELD_PARAMETERS[DEFAULT_ODE_MODEL],
        "field_parameter_values_getter_fn": linear_field_parameter_transform,
    },
    "SOLVERS": {
        "ode_model": ODE_MODELS[DEFAULT_ODE_MODEL],
        "num_nodes": 128,
        "dt": 0.001,
        "tstop": 12.0,
        "solver": (
            "explicit_euler",
            "rush_larsen",
            "generalized_rush_larsen",
            "simplified_implicit_euler",
        ),
        "block_size": 128,
        "field_states": FIELD_STATES[DEFAULT_ODE_MODEL],
        "field_states_getter_fn": get_store_partial_field_state_fn,
        "field_parameters": FIELD_PARAMETERS[DEFAULT_ODE_MODEL],
        "field_parameter_values_getter_fn": linear_field_parameter_transform,
    },
    "ODE MODELS": {
        "ode_model": (
            ODE_MODELS["tentusscher"],
            ODE_MODELS["beeler"],
            ODE_MODELS["grandi"],
        ),
        "num_nodes": 8192,
        "dt": 0.1,
        "tstop": 300.0,
        "solver": "generalized_rush_larsen",
        "field_states": {
            ODE_MODELS["tentusscher"]: FIELD_STATES["tentusscher"],
            ODE_MODELS["beeler"]: FIELD_STATES["beeler"],
            ODE_MODELS["grandi"]: FIELD_STATES["grandi"],
        },
        "field_states_getter_fn": get_store_partial_field_state_fn,
    },
    "TIME STEP": {
        "ode_model": ODE_MODELS[DEFAULT_ODE_MODEL],
        "num_nodes": 8192,
        "dt": (1.0, 0.5, 0.2, 0.1, 0.05),
        "tstop": 300.0,
        "field_states": FIELD_STATES[DEFAULT_ODE_MODEL],
        "field_states_getter_fn": get_store_partial_field_state_fn,
        "field_parameters": FIELD_PARAMETERS[DEFAULT_ODE_MODEL],
        "field_parameter_values_getter_fn": linear_field_parameter_transform,
    },
    "UPDATE HOST/FIELD STATES": {
        "ode_model": ODE_MODELS[DEFAULT_ODE_MODEL],
        "num_nodes": 8192,
        "dt": 0.1,
        "tstop": 300.0,
        "field_states": FIELD_STATES[DEFAULT_ODE_MODEL],
        "field_states_getter_fn": get_store_partial_field_state_fn,
        "field_parameters": FIELD_PARAMETERS[DEFAULT_ODE_MODEL],
        "field_parameter_values_getter_fn": linear_field_parameter_transform,
        "update_host_states": (False, True),
        "update_field_states": (False, True),
    },
    "REPRESENTATION/CSE": {
        "ode_model": ODE_MODELS[DEFAULT_ODE_MODEL],
        "num_nodes": 8192,
        "dt": 0.1,
        "tstop": 300.0,
        "field_states": FIELD_STATES[DEFAULT_ODE_MODEL],
        "field_states_getter_fn": get_store_partial_field_state_fn,
        "field_parameters": FIELD_PARAMETERS[DEFAULT_ODE_MODEL],
        "field_parameter_values_getter_fn": linear_field_parameter_transform,
        "statesrepr": ("named", "array"),
        "paramrepr": ("named", "array", "numerals"),
        "bodyrepr": ("named", "array", "reused_array"),
        "use_cse": (False, True),
    },
    "FAST MATH": {
        "ode_model": ODE_MODELS[DEFAULT_ODE_MODEL],
        "num_nodes": 8192,
        "dt": 0.1,
        "tstop": 300.0,
        "field_states": FIELD_STATES[DEFAULT_ODE_MODEL],
        "field_states_getter_fn": get_store_partial_field_state_fn,
        "field_parameters": FIELD_PARAMETERS[DEFAULT_ODE_MODEL],
        "field_parameter_values_getter_fn": linear_field_parameter_transform,
        "double": (False, True),
        "nvcc_options": (["-ftz=true", "-prec-div=false", "-prec-sqrt=false"], [""]),
    },
    "CUDA CACHE DIR": {
        "ode_model": ODE_MODELS[DEFAULT_ODE_MODEL],
        "num_nodes": 8192,
        "dt": 0.1,
        "tstop": 300.0,
        "field_states": FIELD_STATES[DEFAULT_ODE_MODEL],
        "field_states_getter_fn": get_store_partial_field_state_fn,
        "field_parameters": FIELD_PARAMETERS[DEFAULT_ODE_MODEL],
        "field_parameter_values_getter_fn": linear_field_parameter_transform,
        "cuda_cache_dir": (False, False, None, None),
    },
    "SUBSTEPS": {
        "ode_model": ODE_MODELS[DEFAULT_ODE_MODEL],
        "num_nodes": 8192,
        "dt": 0.1,
        "tstop": 300.0,
        "ode_substeps": (1, 4, 8, 16),
        "field_states": FIELD_STATES[DEFAULT_ODE_MODEL],
        "field_states_getter_fn": get_c_s_partial_field_state_fn,
        "field_parameters": FIELD_PARAMETERS[DEFAULT_ODE_MODEL],
        "field_parameter_values_getter_fn": linear_field_parameter_transform,
    },
}


def getSampleTestSuites(subset=None):
    test_suites = sample_test_suites.copy()

    if subset is None:
        return test_suites
    else:
        if isinstance(subset, str):
            subset = (subset,)
        return {k: test_suites[k] for k in subset}


def runSampleTest(title, **kwargs):
    test_suite = getSampleTestSuites(subset=title)[title].copy()
    test_suite.update(kwargs)
    return runTestSuite(title=title, **test_suite)


def testFloatPrecision(**kwargs):
    return runSampleTest("FLOAT PRECISION", **kwargs)


def testThreadsPerBlock(**kwargs):
    return runSampleTest("THREADS PER BLOCK", **kwargs)


def testNumNodes(**kwargs):
    return runSampleTest("NUM NODES", **kwargs)


def testSolvers(**kwargs):
    return runSampleTest("TIME STEP", **kwargs)


def testOdeModels(**kwargs):
    return runSampleTest("ODE MODELS", **kwargs)


def testTimeStep(**kwargs):
    return runSampleTest("TIME STEP", **kwargs)


def testUpdateStates(**kwargs):
    return runSampleTest("UPDATE HOST/FIELD STATES", **kwargs)


def testRepresentation(**kwargs):
    return runSampleTest("REPRESENTATION/CSE", **kwargs)


def testFastMath(**kwargs):
    return runSampleTest("FAST MATH", **kwargs)


def testCudaCacheDir(**kwargs):
    return runSampleTest("CUDA CACHE DIR", **kwargs)


def testSubsteps(**kwargs):
    return runSampleTest("SUBSTEPS", **kwargs)


def testEverything(subset=None, keep_cuda_code=False, printTimings=True, **kwargs):
    test_suites = getSampleTestSuites(subset=subset).copy()
    test_suites.update(kwargs)
    return runTestSuites(
        test_suites=test_suites,
        keep_cuda_code=keep_cuda_code,
        printTimings=printTimings,
    )


class ThreadedTest(threading.Thread):
    def __init__(self, threadID, testcase, solver, results, threadLock):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.testcase = testcase
        self.solver = solver
        self.results = results
        self.threadLock = threadLock

    def run(self):
        self.threadLock.acquire()

        print "Running test on thread {0}...".format(self.threadID)
        self.results[self.threadID] = runTestCase(
            self.testcase,
            "Thread {0}".format(self.threadID),
            keep_cuda_code=False,
            printTimings=False,
            solver=self.solver,
        )

        self.threadLock.release()


def testSimultaneous(n=4, **kwargs):
    if not ("force" in kwargs and kwargs["force"] == True):
        raise NotImplementedError("testSimultaneous() is broken. Do not run.")

    test_suite = sample_test_suites["FLOAT PRECISION"].copy()
    test_suite["double"] = (True,) * n
    test_suite.update(kwargs)

    testcases, _ = createTestCases(**test_suite)

    threads = list()
    threadLock = threading.Lock()
    results = [None] * n
    for threadID, testcase in enumerate(testcases):
        params = getTestCaseParams(testcase)
        solver = coss.CUDAODESystemSolver(
            testcase.num_nodes,
            testcase.ode,
            init_field_parameters=testcase.field_parameter_values,
            params=params,
        )
        threads.append(ThreadedTest(threadID, testcase, solver, results, threadLock))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    printResults(results)

    return results


def saveResults(title, results, directory):
    if title is None or len(title) == 0:
        print "Title error in saveResults"
        return

    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        print "OSError: {0}".format(e)
        return

    if directory[-1] != os.path.sep:
        directory += os.path.sep

    fname = directory + title.replace(" ", "_")

    print "Writing results to {0}...".format(fname)

    with open(directory + title.replace(" ", "_"), "w") as f:
        f.write("{0}\n".format(title))
        f.write("{0}\n".format(len(results)))
        for i, result in enumerate(results):
            tc = result["testcase"]

            f.write("Test {0} ({1})\n".format(i + 1, result["name"]))

            f.write("{0}\n".format(result["error"]))
            f.write("{0}\n".format(result["runtime"]))
            if result["error"]:
                continue

            f.write("{0}\n".format(tc.block_size))
            f.write("{0}\n".format(tc.bodyrepr))
            f.write("{0}\n".format(tc.cuda_cache_dir))
            f.write("{0}\n".format(tc.double))
            f.write("{0}\n".format(tc.dt))
            if tc.field_parameters is not None:
                f.write(
                    "{0}\n".format(
                        " ".join("{0}".format(fp) for fp in tc.field_parameters)
                    )
                )
            else:
                f.write("\n")
            if tc.field_states is not None:
                f.write(
                    "{0}\n".format(" ".join("{0}".format(fs) for fs in tc.field_states))
                )
            else:
                f.write("\n")
            f.write("{0}\n".format(tc.gpu_arch))
            f.write("{0}\n".format(tc.gpu_code))
            f.write("{0}\n".format(tc.num_nodes))
            f.write("{0}\n".format(tc.nvcc_options))
            f.write("{0}\n".format(tc.ode_model))
            f.write("{0}\n".format(tc.paramrepr))
            f.write("{0}\n".format(tc.solver))
            f.write("{0}\n".format(tc.statesrepr))
            f.write("{0}\n".format(tc.t0))
            f.write("{0}\n".format(tc.tstop))
            f.write("{0}\n".format(tc.update_field_states))
            f.write("{0}\n".format(tc.update_host_states))
            f.write("{0}\n".format(tc.use_cse))

            if tc.field_parameter_values is not None:
                f.write(
                    "{0}\n".format(
                        " ".join("{0}".format(fp) for fp in tc.field_parameter_values)
                    )
                )
            else:
                f.write("\n")
            f.write(
                "{0}\n".format(
                    " ".join("{0}".format(fs) for fs in result["field_states"])
                )
            )
            if tc.stored_field_states is not None:
                f.write("{0}\n".format(len(tc.stored_field_states)))
                f.write(
                    "{0}\n".format(
                        "\n".join(
                            " ".join("{0}".format(fs) for fs in field_states)
                            for field_states in tc.stored_field_states
                        )
                    )
                )
            else:
                f.write("0\n")


def saveResultsSimple(title, results, directory):
    if title is not None or len(title) == 0:
        print "Title error in saveResultsSimple"

    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        print "OSError: {0}".format(e)
        return

    if directory[-1] != os.path.sep:
        directory += os.path.sep

    fname = directory + title.replace(" ", "_")

    data = [title]

    print "Processing results..."

    for i, result in enumerate(results):
        tc = result["testcase"]
        if result["error"]:
            data.append(
                {
                    "i": i,
                    "name": result["name"],
                    "error": result["error"],
                    "runtime": result["runtime"],
                }
            )
            continue
        data.append(
            {
                "i": i,
                "name": result["name"],
                "error": result["error"],
                "runtime": result["runtime"],
                "block_size": tc.block_size,
                "bodyrepr": tc.bodyrepr,
                "cuda_cache_dir": tc.cuda_cache_dir,
                "double": tc.double,
                "dt": tc.dt,
                "field_parameters": tc.field_parameters,
                "field_states": tc.field_states,
                "gpu_arch": tc.gpu_arch,
                "gpu_code": tc.gpu_code,
                "num_nodes": tc.num_nodes,
                "nvcc_options": tc.nvcc_options,
                "ode_model": tc.ode_model,
                "paramrepr": tc.paramrepr,
                "solver": tc.solver,
                "statesrepr": tc.statesrepr,
                "t0": tc.t0,
                "tstop": tc.tstop,
                "update_field_states": tc.update_field_states,
                "update_host_states": tc.update_host_states,
                "use_cse": tc.use_cse,
            }
        )

    print "Writing results to {0}...".format(fname)

    with open(fname, "w") as f:
        pickle.dump(data, f)


def getDataFromFile(_file, get_stored_fstates=False):
    with open(_file) as f:
        title = f.readline().rstrip()
        print "Reading test data '{0}' (may take a while)...".format(title)
        n = int(f.readline().rstrip())
        data = list()
        for _ in xrange(n):
            datum = dict()
            datum["name"] = f.readline().rstrip()
            datum["error"] = f.readline().rstrip()
            datum["runtime"] = float(f.readline().rstrip())

            if datum["error"] != "None":
                print "Encountered result with error ({0}), discarding...".format(name)
                continue

            datum["block_size"] = int(f.readline().rstrip())
            datum["bodyrepr"] = f.readline().rstrip()
            datum["cuda_cache_dir"] = f.readline().rstrip()
            datum["double"] = bool(f.readline().rstrip())
            datum["dt"] = float(f.readline().rstrip())
            datum["field_parameters"] = f.readline().rstrip().split(" ")
            datum["field_states"] = f.readline().rstrip().split(" ")
            datum["gpu_arch"] = f.readline().rstrip()
            datum["gpu_code"] = f.readline().rstrip()
            datum["num_nodes"] = int(f.readline().rstrip())
            datum["nvcc_options"] = f.readline().rstrip()
            datum["ode_model"] = f.readline().rstrip()
            datum["paramrepr"] = f.readline().rstrip()
            datum["solver"] = f.readline().rstrip()
            datum["statesrepr"] = f.readline().rstrip()
            datum["t0"] = float(f.readline().rstrip())
            datum["tstop"] = float(f.readline().rstrip())
            datum["update_field_states"] = bool(f.readline().rstrip())
            datum["update_host_states"] = bool(f.readline().rstrip())
            datum["use_cse"] = bool(f.readline().rstrip())

            datum["field_parameter_values"] = [
                map(float, fpv)
                for fpv in map(
                    ast.literal_eval,
                    [
                        "[" + s.replace("]", " ").replace("[", " ") + "]"
                        for s in f.readline().rstrip().split("] [")
                    ],
                )
            ]
            datum["field_state_values"] = map(float, f.readline().rstrip().split(" "))
            datum["n_stored_fstate_iters"] = int(f.readline().rstrip())
            datum["stored_field_states"] = list()
            if get_stored_fstates:
                for _ in xrange(datum["n_stored_fstate_iters"]):
                    datum["stored_field_states"].append(
                        map(float, f.readline().rstrip().split(" "))
                    )
            else:
                for _ in xrange(datum["n_stored_fstate_iters"]):
                    f.readline()

            data.append(datum)

    return title, data


def getDataFromFileSimple(_file):
    with open(_file, "r") as f:
        raw_data = pickle.load(f)
        title, data = raw_data[0], raw_data[1:]

    return title, data


def _getDefaultPlotConfig(figure):
    config = {
        "style": "ro",
        "plotType": "plot",
        "min_x": None,
        "max_x": None,
        "min_y": None,
        "max_y": None,
        "xticks": None,
        "yticks": None,
    }
    if figure["x"]["type"] == "double":
        config["style"] = None
        config["plotType"] = "bar"
        config["min_y"] = 0
        config["xticks"] = "names"
        return config
    else:
        return config


def plotResults(_file, plotTypes=None, get_stored_fstates=False):
    """
    Example:
    plotTypes=[{'x': {'type': 'block_size'},
                'y': {'type': 'runtime'}},
               {'x': {'type': 'field_parameter_values',
                      'index': 0,
                      'name': 'g_to'},
                'y': {'type': 'field_state_values',
                      'index': 0,
                      'name': 'V'}}]
    """
    title, data = getDataFromFile(_file, get_stored_fstates=get_stored_fstates)
    names = [datum["name"] for datum in data]
    subnames = [name.split(" ", 2)[-1][1:-1] for name in names]
    if plotTypes is None:
        subsubnames = [s.split(",") for s in subnames]
        namekeys, namevalues = zip(
            *[zip(*[s.split("=") for s in ssname]) for ssname in subsubnames]
        )

        if len(set(namekeys)) != 1:
            print "Cannot determine what to plot."
            if len(set(namekeys)) < 16:
                print "Keys", set(namekeys)
            else:
                print len(set(namekeys)), "keys"
            return
        else:
            plotTypes = list()
            for _type in list(set(namekeys))[0]:
                plotTypes.append({"x": {"type": _type}, "y": {"type": "runtime"}})

    validPlotTypes = (
        "block_size",
        "double",
        "dt",
        "field_parameter_values",
        "field_state_values",
        "runtime",
    )
    figures = list()
    for pType in plotTypes:
        if (
            pType["x"]["type"] not in validPlotTypes
            or pType["y"]["type"] not in validPlotTypes
        ):
            print str(pType) + " plot not yet implemented"
        else:
            figures.append(pType)

    for figure in figures:
        fig, ax = plt.subplots()

        plotConfig = _getDefaultPlotConfig(figure)
        plotConfig.update(figure)

        xy_pos = dict()

        for axis in ("x", "y"):
            xy_pos[axis] = None

            if figure[axis]["type"] in ("block_size", "dt", "runtime"):
                xy_pos[axis] = [datum[figure[axis]["type"]] for datum in data]

            if figure[axis]["type"] == "field_state_values":
                xy_pos[axis] = [datum["field_state_values"] for datum in data]

            if figure[axis]["type"] == "field_parameter_values":
                xy_pos[axis] = [datum["field_parameter_values"] for datum in data]

            if figure[axis]["type"] == "double":
                xy_pos[axis] = [i for i in xrange(len(data))]

        x_pos = xy_pos["x"]
        y_pos = xy_pos["y"]

        if plotConfig["plotType"] == "plot":
            if plotConfig["style"] is not None:
                ax.plot(x_pos, y_pos, plotConfig["style"])
            else:
                ax.plot(x_pos, y_pos)
        elif plotConfig["plotType"] == "bar":
            ax.bar(x_pos, y_pos)

        if plotConfig["xticks"] is None:
            min_x, max_x = min(x_pos), max(x_pos)
            if plotConfig["min_x"] is not None:
                min_x = plotConfig["min_x"]
            if plotConfig["max_x"] is not None:
                max_x = plotConfig["max_x"]
            ax.set_xlim((min_x, max_x))
        elif plotConfig["xticks"] == "names":
            plt.xticks(x_pos, names)
        elif plotConfig["xticks"] == "subnames":
            plt.xticks(x_pos, subnames)
        else:
            plt.xticks(x_pos, plotConfig["xticks"])

        if plotConfig["yticks"] is None:
            min_y, max_y = min(y_pos), max(y_pos)
            if plotConfig["min_y"] is not None:
                min_y = plotConfig["min_y"]
            if plotConfig["max_y"] is not None:
                max_y = plotConfig["max_y"]
            ax.set_ylim((min_y, max_y))
        elif plotConfig["yticks"] == "names":
            plt.yticks(y_pos, names)
        elif plotConfig["yticks"] == "subnames":
            plt.yticks(y_pos, subnames)
        else:
            plt.yticks(y_pos, plotConfig["yticks"])

        plt.title(PLOT_TITLES[figure["y"]["type"]].format(title))
        plt.xlabel(PLOT_STRINGS[figure["x"]["type"]])
        plt.ylabel(PLOT_STRINGS[figure["y"]["type"]])

        plt.show()
