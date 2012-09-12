#!/usr/bin/env python
" A script to check the length of the lines in the code"

__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2010 " + __author__
__date__ = "2010-11-26 -- 2012-09-12"
__license__  = "GNU LGPL Version 3.0 or later"


from instant import get_status_output

def list_python_files():
    """
    Return a list of all included src files
    """
    return [f for f in sorted(get_status_output('hg status -c -m -n')[1].split('\n')) \
            if "site-packages" in f and ".py" == f[-3:] and "~" not in f]
