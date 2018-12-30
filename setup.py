#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System imports
from setuptools import setup, Command

from os.path import join as pjoin
import glob
import platform
import sys

# Version number
major = 3
minor = 4

scripts = glob.glob("scripts/*")

requirements = ["sympy<=0.7.7",
                "numpy",
                "scipy",
                "matplotlib",
                "networkx",
                "six",
                "future",
                "modelparameters",
                "instant"]
dependency_links = ['git+https://finsberg@bitbucket.org/finsberg/modelparameters.git#egg=modelparameters']

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

class clean(Command):
    """
    Cleans *.pyc so you should get the same copy as is in the VCS.
    """

    description = "remove build files"
    user_options = [("all","a","the same")]

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        import os
        os.system("utils/clean-files")

class run_tests(Command):
    """
    Runs all tests under the modelparameters/ folder
    """

    description = "run all tests"
    user_options = []  # distutils complains if this is not here.

    def __init__(self, *args):
        self.args = args[0] # so we can pass it to other classes
        Command.__init__(self, *args)

    def initialize_options(self):  # distutils wants this
        pass

    def finalize_options(self):    # this too
        pass

    def run(self):
        import os
        os.system("python utils/run_tests.py")

setup(name = "gotran",
      version = "{0}.{1}".format(major, minor),
      description = """
      A declarative language describing ordinary differential equations.
      """,
      author = "Henrik Finsberg",
      author_email = "henriknf@simula.no",
      packages = ["gotran", "gotran.common", "gotran.model",
                  "gotran.algorithms", "gotran.codegeneration",
                  "gotran.input", "gotran.solver"],
      install_requires=requirements,
      dependency_links=dependency_links,
      scripts = scripts,
      cmdclass    = {'test': run_tests,
                     'clean': clean,
                     },

      )
