#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2010 " + __author__
__date__ = "2012-02-22 -- 2012-08-23"
__license__  = "GNU LGPL Version 3.0 or later"


# System imports
from distutils.core import setup
from os.path import join as pjoin
import glob
import platform
import sys

# Version number
major = 2
minor = 0

scripts = [pjoin("scripts", "gotran2")]

if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    batch_files = []
    for script in scripts:
        batch_file = script + ".bat"
        f = open(batch_file, "w")
        f.write('python "%%~dp0\%s" %%*\n' % split(script)[1])
        f.close()
        batch_files.append(batch_file)
    scripts.extend(batch_files)

setup(name = "Gotran2",
      version = "{0}.{1}".format(major, minor),
      description = """
      A declarative language describing ordinary differential equations.
      """,
      author = __author__.split("(")[0],
      author_email = __author__.split("(")[1][:-1],
      packages = ["gotran2", "gotran2.common", "gotran2.model",
                  "gotran2.algorithms", "gotran2.codegeneration"],
      package_dir = {"gotran2": "src"},
      scripts = scripts,
      )
