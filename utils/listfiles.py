#!/usr/bin/env python
" A script to check the length of the lines in the code"

__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2010 " + __author__
__date__ = "2010-11-26 -- 2012-02-23"
__license__  = "GNU LGPL Version 3.0 or later"


from gotran2.common.commands import get_output

def list_files():
    """
    Return a list of all included src files
    """
    return [f for f in get_output('hg status -c -m -n').split('\n') if "src" in f and ".py" in f and "~" not in f]

