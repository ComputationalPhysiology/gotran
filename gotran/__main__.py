import sys

from . import scripts

__doc__ = """gotran (General Ode TRANslator)
cellml2gotran
gotran2c
gotran2cpp
gotran2cuda
gotran2dolfin
gotran2julia
gotran2latex
gotran2matlab
gotran2opencl
gotran2py
gotranexport
gotranprob
gotranrun
gotran2md
"""


def main():

    if len(sys.argv) < 2:
        print(__doc__)
        return

    # Show help message
    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(__doc__)
    else:
        sys.argv = sys.argv[1:]

        if sys.argv[0] == "cellml2gotran":
            scripts.cellml2gotran.main()
        elif sys.argv[0] == "gotran2c":
            scripts.gotran2c.main()
        elif sys.argv[0] == "gotran2cpp":
            scripts.gotran2cpp.main()
        elif sys.argv[0] == "gotran2cuda":
            scripts.gotran2cuda.main()
        elif sys.argv[0] == "gotran2dolfin":
            scripts.gotran2dolfin.main()
        elif sys.argv[0] == "gotran2julia":
            scripts.gotran2julia.main()
        elif sys.argv[0] == "gotran2latex":
            scripts.gotran2latex.main()
        elif sys.argv[0] == "gotran2matlab":
            scripts.gotran2matlab.main()
        elif sys.argv[0] == "gotran2opencl":
            scripts.gotran2opencl.main()
        elif sys.argv[0] == "gotran2py":
            scripts.gotran2py.main()
        elif sys.argv[0] == "gotranexport":
            scripts.gotranexport.main()
        elif sys.argv[0] == "gotranprobe":
            scripts.gotranprobe.main()
        elif sys.argv[0] == "gotranrun":
            scripts.gotranrun.main()
        elif sys.argv[0] == "gotran2md":
            scripts.gotran2md.main()
        else:
            print(__doc__)
            print(f"Unknown argument {sys.argv[0]}")


if __name__ == "__main__":
    main()
