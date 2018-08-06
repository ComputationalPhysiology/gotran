# Gotran - General ODE TRAnslator


Gotran:

  - provides a Python interface to declare arbitrary ODEs.

  - provides an interface for generating CUDA/C/C++/Python/Matlab code for
    a number of functions including the right hand side and symbolic
    generation of a Jacobian.

  - is intentionally lightweight, and could be interfaced by other
    Python libraries needing functionalities to abstract a general
    ODE.

  - depends on NumPy, and on SymPy. See further instructions in
    INSTALL

  - can load models from external ODE desciption files such as CellML


Source code can be found at <https://bitbucket.org/finsberg/gotran>
See the [installation instructions](INSTALL.md) for details on how to install Gotran.

## License

gotran is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

gotran is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with gotran. If not, see <http://www.gnu.org/licenses/>.


## Contributors
Gotran is developed by Johan Hake.
The version of gotran found in this repository is maintained by Henrik Finsberg.
Issues can be reported to <henriknf@simula.no>
