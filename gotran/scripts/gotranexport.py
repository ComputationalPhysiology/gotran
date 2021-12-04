#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Johan Hake (hake.dev@gmail.com)"
__date__ = "2013-05-07 -- 2015-06-04"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU LGPL Version 3.0 or later"

from modelparameters.parameterdict import ParameterDict
from modelparameters.parameters import Param
from modelparameters.logger import error
from gotran.model.loadmodel import load_ode
from gotran.model.ode import ODE


def gotranexport(filename, params):

    if len(params.components) == 1 and params.components[0] == "":
        error("Expected at least 1 component.")

    if params.name == "":
        error("Expected a name of the exported ODE.")

    if params.name in params.components:
        error("Expected the name of the ODE to be different from the components.")

    sub_ode = ODE(params.name)
    sub_ode.import_ode(load_ode(filename), components=params.components)
    sub_ode.finalize()
    sub_ode.save()


def main():
    import os
    import sys

    params = ParameterDict(
        components=Param(
            [""],
            description="List all components that will be " "exported.",
        ),
        name=Param("", description="Specify name of exported ODE."),
    )
    params.parse_args(usage="usage: %prog FILE [options]")

    if len(sys.argv) < 2:
        raise RuntimeError("Expected a single gotran file argument")

    if not os.path.isfile(sys.argv[1]):
        raise IOError("Expected the argument to be a file")

    file_name = sys.argv[1]
    gotranexport(file_name, params)


if __name__ == "__main__":
    main()
