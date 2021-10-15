[![CI](https://github.com/ComputationalPhysiology/gotran/actions/workflows/main.yml/badge.svg)](https://github.com/ComputationalPhysiology/gotran/actions/workflows/main.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ComputationalPhysiology/gotran/master.svg)](https://results.pre-commit.ci/latest/github/ComputationalPhysiology/gotran/master)
[![codecov](https://codecov.io/gh/ComputationalPhysiology/gotran/branch/master/graph/badge.svg?token=jfiLeIpkNa)](https://codecov.io/gh/ComputationalPhysiology/gotran)

# Gotran - General ODE TRAnslator

Gotran:

- provides a Python interface to declare arbitrary ODEs.

- provides an interface for generating CUDA/C/C++/OpenCL/Julia/Python/Matlab code for
  a number of functions including the right hand side and symbolic
  generation of a Jacobian.

- is intentionally lightweight, and could be interfaced by other
  Python libraries needing functionalities to abstract a general
  ODE.

- depends on NumPy, and on SymPy. See further instructions in
  INSTALL

- can load models from external ODE description files such as CellML

## Install

You can install gotran through pip

```
python -m pip install gotran
```

or conda

```
conda install -c comphy gotran
```

or if you want to work with the latest version you can install the developement version by either cloning the repository and run

```
python -m pip install .
```

inside the root directory or run

```
python -m pip install git+https://github.com/ComputationalPhysiology/gotran.git
```

## Documentation

Documentation can be found here: <http://computationalphysiology.github.io/gotran/>

## Source code

Source code can be found at <https://github.com/ComputationalPhysiology/gotran>.

## License

gotran is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

gotran is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with gotran. If not, see <http://www.gnu.org/licenses/>.

## Contributors

Gotran is developed by Johan Hake.
The version of gotran found in this repository is maintained by Henrik Finsberg and Kristian Hustad.

```

```
