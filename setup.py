#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System imports
from setuptools import setup, Command
from shutil import rmtree
import os
import io
from os.path import join as pjoin
import glob
import platform
import sys

# Version number
major = 2020
minor = 1.0
VERSION = "{0}.{1}".format(major, minor)

DESCRIPTION = "A declarative language describing ordinary differential equations."

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


scripts = glob.glob("scripts/*")

requirements = [
    "sympy<=1.1.1",
    "numpy",
    "scipy",
    "matplotlib",
    "networkx",
    "six",
    "future",
    "modelparameters",
    "instant",
]

if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    batch_files = []
    for script in scripts:
        batch_file = script + ".bat"
        with open(batch_file, "w") as f:
            f.write('python "%%~dp0\%s" %%*\n' % os.path.split(script)[1])
        batch_files.append(batch_file)
    scripts.extend(batch_files)


class clean(Command):
    """
    Cleans *.pyc so you should get the same copy as is in the VCS.
    """

    description = "remove build files"
    user_options = [("all", "a", "the same")]

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
        self.args = args[0]  # so we can pass it to other classes
        Command.__init__(self, *args)

    def initialize_options(self):  # distutils wants this
        pass

    def finalize_options(self):  # this too
        pass

    def run(self):
        import os

        os.system("python utils/run_tests.py")


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("python -m twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(VERSION))
        os.system("git push --tags")

        sys.exit()


setup(
    name="gotran",
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ComputationalPhysiology/gotran",
    author="Johan Hake and Henrik Finsberg",
    author_email="henriknf@simula.no",
    packages=[
        "gotran",
        "gotran.common",
        "gotran.model",
        "gotran.algorithms",
        "gotran.codegeneration",
        "gotran.input",
        "gotran.solver",
    ],
    install_requires=requirements,
    scripts=scripts,
    cmdclass={"test": run_tests, "clean": clean, "upload": UploadCommand,},
)
