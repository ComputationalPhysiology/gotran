"Run all tests"

__author__ = "Johan Hake (hake.dev@gmail.com)"
__date__ = "2012-08-15 -- 2014-01-31"
__copyright__ = "Copyright (C) 2012 " + __author__
__license__  = "GNU LGPL version 3.0"

from modelparameters.commands import get_status_output

import re, sys, os

failed = []

num_tests = 0
timing = 0.0

# Run tests
root_dir = os.path.abspath(os.path.curdir)
for dirpath, dirnames, filenames in os.walk("gotran"):
    if os.path.basename(dirpath) == "test":
        
        print("")
        print("Running tests in: %s" % dirpath)
        print("-"*79)
        for test in filenames:
            if not re.findall("test_(\w+).py", test):
                continue
            os.chdir(os.path.join(root_dir, dirpath))
            fail, output = get_status_output("python %s" % test)
            num_tests += int(re.findall("Ran (\d+) tests", output)[0])
            timing += float(re.findall("tests in ([\d\.]+)s", output)[0])
            if fail:
                failed.append(output)

            os.chdir(root_dir)
print()
print("-"*79)
print("Ran %d tests in %.3fs" % (num_tests, timing))
if failed:
    for output in failed:
        print(output)

sys.exit(len(failed))
