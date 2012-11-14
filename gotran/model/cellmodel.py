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

# FIXME: This file is depricated...

__all__ = ["CellModel", "gccm"]

from gotran.common.utils import *

# Holder for all cellmodels
_all_cellmodels = {}

# Holder for current CellModel
_current_cellmodel = None

# FIXME: Consider adding a more general class ODE to hold all logic for ODEs
# FIXME: then subclass ODE to create CellModel. Then we can include alot of
# FIXME: interesting information about a CellModel without mixing that stuff
# FIXME: with the actuall ODE.
class CellModel(object):
    """
    Basic class for storying information of a cell model
    """
    def __new__(cls, name):
        """
        Create a CellModel instance.
        """
        check_arg(name, str, 0)

        # If the CellModel already excist just return the instance
        if name in _all_cellmodels:
            return _all_cellmodels[name]

        # Create object
        return object.__new__(cls)
        
    def __init__(self, name):
        """
        Initialize a CellModel
        """

        # Set current CellModel
        _current_cellmodel = self

        # Do not reinitialized object if it already excists
        if name in _all_cellmodels:
            return

        # Initialize attributes
        self.name = name
        self.state_symbols = {}
        self.field_state_symbols = {}
        self.state_values = {}
        self.parameter_symbols = {}
        self.parameter_values = {}
        self.state_derivatives = {}
        self.all_variables = []

        # Store instance for future lookups
        _all_cellmodels[name] = self

# Construct a default CellModel
_current_cellmodel = CellModel("Default")
        
def gccm():
    """
    Return the current CellModel
    """
    assert(isinstance(_current_cellmodel, CellModel))
    return _current_cellmodel
    
