import subprocess as sp
from pathlib import Path

import pytest

here = Path(__file__).parent.absolute()

# cellml2gotran
# ("gotran2matlab", ".m")<

# ("gotranexport", ".h")
# ("gotranprob", ".h")
# ("gotranrun", ".h")


@pytest.mark.parametrize(
    "script, ext",
    [
        ("gotran2c", ".h"),
        ("gotran2cpp", ".h"),
        ("gotran2cuda", ".cu"),
        ("gotran2dolfin", ".py"),
        ("gotran2julia", ".jl"),
        ("gotran2latex", ".tex"),
        ("gotran2opencl", ".cl"),
        ("gotran2py", ".py"),
    ],
)
def test_gotran_codegen_scripts(script, ext):
    modelname = here.joinpath("model").joinpath("tentusscher_2004_mcell_updated")
    args = ["python3", "-m", "gotran", script, f"{modelname}.ode"]
    print(" ".join(args))
    output = sp.run(
        [
            "python3",
            "-m",
            "gotran",
            script,
            modelname.with_suffix(".ode").as_posix(),
        ],
        cwd=here,
    )
    output.check_returncode()
    outfile = modelname.with_suffix(ext)
    assert outfile.is_file()
    outfile.unlink()
