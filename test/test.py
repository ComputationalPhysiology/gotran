"Run all tests"

__author__ = "Johan Hake (hake.dev@gmail.com)"
__date__ = "2012-05-07 -- 2012-05-08"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU LGPL version 3.0"

# System imports
import sys
import os

# Gotran imports
from gotran2.common.commands import get_status_output

pwd = os.path.dirname(os.path.abspath(__file__))

# Tests to run
tests = ["unit", "regression"]

failed = []

# Command to run
command = "python test.py " + " ".join(sys.argv[1:])

# Run testsg
for test in tests:
    if not os.path.isfile(os.path.join(test, "test.py")):
        continue
    print ""
    print "Running tests: %s" % test
    print "----------------------------------------------------------------------"
    os.chdir(os.path.join(pwd, test))
    fail, output = get_status_output(command)
    print output
    if fail:
        failed.append(fail)

sys.exit(len(failed))
