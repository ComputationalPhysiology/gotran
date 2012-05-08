"""Run all unit tests."""

__author__ = "Johan Hake (hake.dev@gmail.com)"
__date__ = "2010-03-19 -- 2012-05-08"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU LGPL version 3.0"

# System imports
import sys
import os
import re

# Gotran imports
from gotran2.common.commands import get_output

# Tests to run
tests = dict(ode = ["panfilov"])

# Run tests
failed = []
for test_dir, tests in tests.items():
    print ""
    print "Running unit tests for %s" % test_dir
    print "----------------------------------------------------------------------"

    for test in tests:
        output = get_output("cd {0} && python {1}.py".format(test_dir, test))
        if "OK" in output:
            num_tests = int(re.search("Ran (\d+) test", output).groups()[0])
            print "{0}: OK ({1} tests)".format(test, num_tests)
        else:
            print "{0}: *** Failed".format(test)
            failed += [(test, output)]

# Print output for failed tests
for (test, output) in failed:
    print "One or more unit tests failed for {0}:".format(test)
    print output

# Return error code if tests failed
sys.exit(len(failed) != 0)
