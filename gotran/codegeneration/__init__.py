# Copyright (C) 2012 Johan Hake
#
# This file is part of Gotran.
#
# Gotran is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Gotran is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Gotran. If not, see <http://www.gnu.org/licenses/>.

# Import gotran modules
from . import codegenerators
from . import compilemodule
from . import latexcodegenerator
from . import codecomponent
from . import algorithmcomponents
from . import solvercomponents

# Import classes and routines from gotran modules
from .codecomponent import *
from .algorithmcomponents import *
from .solvercomponents import *
from .codegenerators import *
from .latexcodegenerator import *
from .latexcodegenerator import _default_latex_params
from .compilemodule import compile_module


# Assign the __all__ attribute
__all__ = [name for name in list(globals().keys()) if name[:1] != "_"]

