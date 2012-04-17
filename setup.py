#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2010 " + __author__
__date__ = "2012-02-22 -- 2012-02-23"
__license__  = "GNU LGPL Version 3.0 or later"


# System imports
from distutils.core import setup
from os.path import join as pjoin
import glob

# Version number
major = 2
minor = 0

setup(name = "Gotran2",
      version = "%d.%d" % (major, minor),
      description = """
      A declarative language describing ordinary differential equations.
      """,
      author = __author__.split("(")[0],
      author_email = __author__.split("(")[1][:-1],
      packages = ["gotran2", "gotran2.common", "gotran2.models"],
      package_dir = {"gotran2": "src",
                     "gotran2.common":pjoin("src","common"),
                     "gotran2.common":pjoin("src","models"),
                     })
