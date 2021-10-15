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

__all__ = ["load", "save", "present_time_str"]

import glob
import os
import re

# System imports
import time
from pickle import Pickler, Unpickler

import numpy
from modelparameters.logger import INFO, info, set_default_exception, set_log_level
from modelparameters.utils import check_arg

TIME_FORMAT = "%Y.%m.%d-%H.%M.%S"

set_log_level(INFO)


class GotranException(RuntimeError):
    "Base class for ModelParameters exceptions"
    pass


set_default_exception(GotranException)


def present_time_str():
    "Returns the present time nicely formated"
    return time.strftime(TIME_FORMAT)


def save(basename, **data):
    """
    Save data using cPickle

    @type basename : str
    @param basename : The name of the file to save the data in, .cpickle
    will be appended to the file name if not provided
    @param data : The actuall data to be saved.

    """

    check_arg(basename, str, 0)

    # If zero data size just return
    if len(data) == 0:
        return

    filename = basename if ".cpickle" in basename else basename + ".cpickle"

    f = open(filename, "w")

    p = Pickler(f)

    # Dump the dictionary kwarg
    p.dump(data)
    f.close()


def compare_dicts(p1, p2):
    "Recursively compares a dict of values"
    assert isinstance(p1, dict)
    assert isinstance(p2, dict)
    try:
        ret = p1 == p2
    except ValueError:
        ret = True
        for key, value in p1.items():
            if isinstance(value, numpy.ndarray):
                ret = ret and (value == p2[key]).all()
            elif isinstance(value, dict):
                ret = ret and compare_dicts(value, p2[key])
            else:
                ret = ret and value == p2[key]
    return ret


def load(basename, latest_timestamp=False, collect=False):
    """
    Load data using cPickle

    @type basename : str
    @param basename : The name of the file where the data will be loaded from,
    '.cpickle' will be appended to the file name if not provided
    @type latest_timestamp : bool
    @param latest_timestamp : If true return the data from latest version of
    saved data with the same basename
    @type collect : bool
    @param collect : If True collect all data with the same basename and
    the same parameters
    """

    check_arg(basename, str, 0)

    if latest_timestamp and collect:
        raise TypeError("'collect' and 'latest_timestamp' cannot both be True")

    # If not collect just return the data froma single data file
    if not collect:
        return load_single_data(basename, latest_timestamp)

    filenames = get_data_filenames(basename)

    # No filenames with timestamp. Try to return data file without timestamp
    if not filenames:
        return load_single_data(basename, False)

    # Start with the latest filename and load the data and collect them if
    # the data have the same parameter
    data = load_single_data(filenames.pop(-1), False)
    params = data["params"]
    for filename in reversed(filenames):
        local_data = load_single_data(filename, False)
        if not compare_dicts(params, local_data["params"]):
            info("Not the same parameters, skipping data from '%s'", filename)
            continue
        merge_data_dicts(data, local_data)

    return data


def merge_data_dicts(data0, data1):
    "Merge data from data1 into data0"

    def recursively_merge_data(data0, data1):
        for (key, values), org_values in zip(iter(data1.items()), list(data0.values())):
            if isinstance(values, dict):
                data0[key] = recursively_merge_data(org_values, values)
            elif isinstance(values, list):
                if isinstance(values[0], list):
                    for i in range(len(values)):
                        org_values[i].extend(values[i])
                else:
                    org_values.extend(values)
                data0[key] = org_values
        return data0

    for (key, values), org_values in zip(iter(data1.items()), list(data0.values())):
        if key == "params":
            continue

        data0[key] = recursively_merge_data(values, org_values)


def load_single_data(basename, latest_timestamp):
    "Helper function for load"

    if latest_timestamp:
        filenames = get_data_filenames(basename)
        if not filenames:
            raise IOError(f"No files with timestamp for basename: '{basename}' excist")
        basename = filenames[-1]

    filename = basename if ".cpickle" in basename else basename + ".cpickle"

    if not os.path.isfile(filename):
        raise IOError(f"No file with basename: '{basename}' excists")

    info("Loading data from: %s", filename)
    f = open(filename, "r")

    return Unpickler(f).load()


def get_data_filenames(basename):
    "Helper functions for getting data filenames"

    basename = basename if ".cpickle" in basename else basename.replace(".cpickle", "")
    pattern = re.compile(
        f"{basename}-[0-9]+\\.[0-9]+\\.[0-9]+-[0-9]+\\.[0-9]+\\.[0-9]+.cpickle",
    )

    filenames = [
        filename
        for filename in glob.glob(f"{basename}*.cpickle")
        if re.search(pattern, filename)
    ]

    if not filenames:
        return []

    filenames.sort()
    return filenames
