.. gotran documentation master file, created by
   sphinx-quickstart on Thu Sep  6 12:50:29 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gotran - General ODE TRAnslator
===============================

Gotran:

* provides a Python interface to declare arbitrary ODEs.
* provides an interface for generating CUDA/C/C++/Python/Matlab code
  for a number of functions including the right hand side and symbolic
  generation of a Jacobian. 
* is intentionally lightweight, and could be interfaced by other
  Python libraries needing functionalities to abstract a general ODE. 
* depends on NumPy, and on SymPy. See further instructions in ...
* can load models from external ODE desciption files such as CellML


Source code can be found at https://bitbucket.org/finsberg/gotran See
the installation instructions for details on how to install Gotran.


Installation
------------
You can install gotran with either with `pip
<https://pypi.org/project/gotran/>`_:

.. code:: shell

    pip install gotran

or if you want the latest features, you can install from `source
<https://bitbucket.org/finsberg/gotran>`_:

.. code:: shell

    pip install git+https://finsberg@bitbucket.org/finsberg/gotran.git


Source code
-----------
Gotran is orginally developed by Johan Hake, and the original
source code can be found in his `repoistory
<https://bitbucket.org/johanhake/gotran>`_.
The current maintained version can be foud `here
<https://bitbucket.org/finsberg/gotran>`_.


Contributors
------------
The main contributors are

* Henrik Finsberg (henriknf@simula.no)

* Johan Hake (hake.dev@gmail.com)


License
-------
GNU GPLv3


.. toctree::
   :maxdepth: 1
   :caption: Programmers reference:

   gotran




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
