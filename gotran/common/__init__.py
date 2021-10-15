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
# from modelparameters.logger import set_default_exception, set_log_level, INFO
# import modelparameters.commands as commands
from .dicts import adict
from .dicts import odict
from .disk import GotranException
from .disk import load
from .disk import present_time_str
from .disk import save
from .options import parameters

# ModelParameters imports

# Base class for ModelParameters exceptions

# from modelparameters.utils import *

# from modelparameters.utils import (Range, Timer, list_types, clear_timings, tic,
#                              toc, is_iterable, add_iterable, camel_capitalize,
#                              tuplewrap, listwrap, check_arg, check_arginlist,
#                              check_kwarg, quote_join, deprecated, format_time,
#                              value_formatter, param2value)
# # gotran imports


# Set initial log level to INFO


__all__ = [
    "adict",
    "odict",
    "load",
    "save",
    "present_time_str",
    "parameters",
    "GotranException",
]
# __all__ = [_name for _name in list(globals().keys()) if _name[0] != "_"]
# __all__ = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "Logger", "commands",
#            "Range", "Timer", "list_types", "clear_timings", "tic",
#            "toc", "is_iterable", "add_iterable", "camel_capitalize",
#            "tuplewrap", "listwrap", "check_arg", "check_arginlist",
#            "check_kwarg", "quote_join", "deprecated", "format_time",
#            "value_formatter", "param2value", "adict", "odict", "load", "save",
#            "present_time_str", "parameters"]
