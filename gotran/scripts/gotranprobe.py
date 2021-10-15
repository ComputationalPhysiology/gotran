#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Johan Hake (hake.dev@gmail.com)"
__date__ = "2013-05-07 -- 2015-06-04"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU LGPL Version 3.0 or later"
import os
import sys
from collections import deque

from modelparameters.logger import INFO, WARNING, set_log_level
from modelparameters.parameterdict import ParameterDict
from modelparameters.parameters import Param

from gotran.model.loadmodel import load_ode
from gotran.model.odeobjects import Comment


def join_indent_and_split(code, baseindent=4):

    code = deque(code)
    output = [" " * baseindent + code.popleft()] if code else [""]
    while code:
        if len(output[-1]) + 2 + len(code[0]) > 80:
            output.append(" " * baseindent + code.popleft())
        else:
            output[-1] += ", " + code.popleft()

    return "\n".join(output)


def gotranprobe(filename, params):

    set_log_level(WARNING)
    ode = load_ode(filename)
    set_log_level(INFO)

    if params.flat_view:
        print()
        print(f"  Components({len(ode.components)}):")
        print(join_indent_and_split([comp.name for comp in ode.components]))
        print()
        print(f"  States({len(ode.full_states)}):")
        print(join_indent_and_split([state.name for state in ode.full_states]))
        if len(ode.parameters) > 0:
            print()
            print(f"  Parameters({len(ode.parameters)}):")
        print(join_indent_and_split([param.name for param in ode.parameters]))
        print()
        print(f"  Intermediates({len(ode.intermediates)}): ")
        print(join_indent_and_split([interm.name for interm in ode.intermediates]))
    else:
        for comp in ode.components:
            if comp.states or comp.parameters or comp.intermediates:
                print(f"\n  {comp.name}:")
                if comp.full_states:
                    print(
                        "    States:        ",
                        ", ".join(state.name for state in comp.full_states),
                    )
                if comp.parameters:
                    print(
                        "    Parameters:    ",
                        ", ".join(param.name for param in comp.parameters),
                    )
                if comp.intermediates:
                    print(
                        "    Intermediates: ",
                        ", ".join(interm.name for interm in comp.intermediates),
                    )
                    print(
                        "    Dependencies:  ",
                        ", ".join(
                            set(
                                dep.name
                                for interm in comp.intermediates
                                for dep in ode.expression_dependencies[interm]
                                if not isinstance(dep, Comment)
                                and dep not in comp.full_states + comp.parameters
                            ),
                        ),
                    )


def main():

    params = ParameterDict(
        flat_view=Param(True, description="List all objects in a flat view"),
    )
    params.parse_args(usage="usage: %prog FILE [options]")

    if len(sys.argv) < 2:
        raise RuntimeError("Expected a single gotran file argument")

    if not os.path.isfile(sys.argv[1]):
        raise IOError("Expected the argument to be a file")

    file_name = sys.argv[1]
    gotranprobe(file_name, params)


if __name__ == "__main__":
    main()
