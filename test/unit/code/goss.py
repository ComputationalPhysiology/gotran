__author__ = "Johan Hake (hake.dev@gmail.com)"
__date__ = "2012-09-20 -- 2012-09-20"
__copyright__ = "Copyright (C) 2012 " + __author__
__license__  = "GNU LGPL Version 3.0 or later"

from gotran import *
from gotran.codegeneration.gosscodegenerator import GossCodeGenerator
from gotran.codegeneration.codegenerator import ODERepresentation

ode = load_ode("winslow")
oderepr = ODERepresentation(ode, keep_intermediates=True, use_cse=False)
gossgen = GossCodeGenerator(oderepr)

print gossgen.generate()
