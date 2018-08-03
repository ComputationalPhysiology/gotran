# Copyright (C) 2011-2012 Johan Hake
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

# ModelParameters imports
from modelparameters.logger import *
import modelparameters.commands as commands

# Base class for ModelParameters exceptions
class GotranException(RuntimeError):
    "Base class for ModelParameters exceptions"
    pass

set_default_exception(GotranException)
from modelparameters.utils import *

# gotran imports
from gotran.common.dicts import *
from gotran.common.disk import *
from gotran.common.options import *

# Set initial log level to INFO
set_log_level(INFO)

__all__ = [_name for _name in list(globals().keys()) if _name[0] != "_"]
