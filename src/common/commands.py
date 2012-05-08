__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2010 " + __author__
__date__ = "2012-02-26 -- 2012-05-07"
__license__  = "GNU LGPL Version 3.0 or later"

from subprocess import Popen, PIPE, STDOUT

__all__ = ['get_output', 'get_status_output', 'get_status_output_errors']

def get_status_output_errors(cmd, input=None, cwd=None, env=None):
    pipe = Popen(cmd, shell=True, cwd=cwd, env=env, stdout=PIPE, stderr=PIPE)

    (output, errout) = pipe.communicate(input=input)

    status = pipe.returncode

    return (status, output, errout)

def get_status_output(cmd, input=None, cwd=None, env=None):
    pipe = Popen(cmd, shell=True, cwd=cwd, env=env, stdout=PIPE, stderr=STDOUT)

    (output, errout) = pipe.communicate(input=input)
    assert not errout

    status = pipe.returncode

    return (status, output)

def get_output(cmd, input=None, cwd=None, env=None):
    return get_status_output(cmd, input, cwd, env)[1]
