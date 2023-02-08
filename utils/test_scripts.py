"""
Run test for all scripts
"""
import os
import time

from modelparameters.commands import get_status_output

root_dir = os.path.abspath(os.path.curdir)
curdir = os.path.abspath(os.path.dirname(__file__))

failed_output = []
failed = 0
num_tests = 0

cmd2file = {
    "gotran2py": ["tentusscher_2004_mcell_updated.py"],
    "gotran2c": ["tentusscher_2004_mcell_updated.h"],
    "gotran2dolfin": ["tentusscher_2004_mcell_updated.py"],
    "gotran2julia": ["tentusscher_2004_mcell_updated.jl"],
    "gotran2cpp": ["tentusscher_2004_mcell_updated.h"],
    "gotran2cuda": ["tentusscher_2004_mcell_updated.cu"],
    "gotran2latex": ["tentusscher_2004_mcell_updated.tex"],
    "gotran2matlab": [
        "tentusscher_2004_mcell_updated_init_parameters.m",
        "tentusscher_2004_mcell_updated_init_states.m",
        "tentusscher_2004_mcell_updated_monitor.m",
        "tentusscher_2004_mcell_updated_monitored_names.m",
        "tentusscher_2004_mcell_updated_rhs.m",
    ],
    "cellml2gotran": ["tentusscher_2004_mcell.ode"],
}
commands = [
    "gotranrun",
    "gotranprobe",
    "gotran2c",
    "gotran2cpp",
    "gotran2cuda",
    "gotran2latex",
    "gotran2matlab",
    "gotran2dolfin",
    "gotran2julia",
    "gotran2py",
    "cellml2gotran",
]

all_output = []

os.chdir(curdir)

start = time.time()
for command in commands:
    print(f"Test scripts {command}")
    if command == "cellml2gotran":
        infile = "tentusscher_noble_noble_panfilov_2004_a.cellml"
    else:
        infile = "tentusscher_2004_mcell_updated.ode"
    cmd = f"{command} {infile}"
    if command == "gotranrun":
        cmd += " --plot_y"

    fail, output = get_status_output(cmd)
    num_tests += 1

    if fail:
        failed = 1
        failed_output.append(output)

    if command in cmd2file:
        for f in cmd2file[command]:
            if not os.path.isfile(f):
                failed = 1
                msg = ("Command {} failed: " "File {} does not exist.").format(
                    command,
                    f,
                )
                failed_output.append(msg)
            else:
                os.remove(f)

end = time.time()
print(f"Ran {num_tests} tests in {end - start:.2f} s")
if failed:
    for output in failed_output:
        print(output)
else:
    print("All tests passed")

os.chdir(root_dir)
