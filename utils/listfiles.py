#!/usr/bin/env python
" A script to check the length of the lines in the code"

# Copyright (C) 2011-2014 Johan Hake
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

from instant import get_status_output


def list_python_files():
    """
    Return a list of all included src files
    """
    return [
        f
        for f in sorted(
            get_status_output("git ls-tree --name-only --full-tree -r HEAD")[1].split(
                "\n",
            ),
        )
        if ".py" == f[-3:]
    ]
