#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import os

from gotran.input.cellml import CellMLParser

# Gotran imports


def define_parser(change_state_names):
    """
    Return a populated option parser
    """
    import optparse

    parser = optparse.OptionParser(
        usage="%prog [options]",
        description="Translate CellML files into gotran files.",
    )

    def list_parser(option, opt_str, value, parser, arg_list):
        rargs = parser.rargs
        # print rargs
        while rargs:
            arg = rargs[0]

            # Stop if we hit an arg like "--par", i.e, PAR_PREFIX
            if arg[0] == "-":
                break
            else:
                arg_list.append(rargs.pop(0))

    parser.add_option(
        "-c",
        "--change_state_names",
        help="Change the name locally of a state name. " "Usage: -c R",
        action="callback",
        callback=list_parser,
        callback_args=(change_state_names,),
        metavar="EQN1 EQN",
    )

    parser.add_option(
        "-s",
        "--use_sympy_integers",
        help="If yes dedicated sympy "
        "integers will be used instead of the "
        "integers 0-10. This will turn 1/2 into a "
        "sympy rational instead of the float 0.5.",
        action="store_true",
        default=False,
        dest="use_sympy_integers",
    )

    parser.add_option(
        "-p",
        "--strip_parent_name",
        help="If True strip the name from "
        "the child component it contains the name of the "
        "parent component.",
        action="store_true",
        default=True,
        dest="strip_parent_name",
    )

    parser.add_option(
        "-o",
        "--output",
        help="""Name of output file. If not provided
        then the same name as the name of the cellml file will be used""",
        default="",
        type=str,
        dest="output",
    )
    return parser


def main():

    change_state_names = []
    parser = define_parser(change_state_names)

    options, args = parser.parse_args()

    param = CellMLParser.default_parameters()

    for name in param:
        if hasattr(options, name):
            param[name] = getattr(options, name)

    if len(args) != 1:
        print(args)
        raise ValueError("Expected 1 and only one CellML file.")

    cellml = CellMLParser(args[0], param)
    gotran_code = cellml.to_gotran()

    if options.output == "":
        output = cellml.name
    else:
        output = os.path.splitext(options.output)[0]

    open(f"{output}.ode", "w").write(gotran_code)


if __name__ == "__main__":
    main()
