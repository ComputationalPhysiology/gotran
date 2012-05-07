__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2010 " + __author__
__date__ = "2010-09-22 -- 2012-04-18"
__license__  = "GNU LGPL Version 3.0 or later"

# System imports
import time as _time
import math as _math

# gotran2 imports
from gotran2.common.logger import *

# Define scalar
scalar = (int, float)

_toc_time = 0.0

def _floor(value):
    return int(_math.floor(value))

def format_time(time):
    """
    Return a formated version of the time argument

    @type time: float
    @param time: A time in seconds
    """
    minutes = _floor(time/60)
    seconds = _floor(time%60)
    if minutes < 1:
        return "%d s"%seconds
    hours = _floor(minutes/60)
    minutes = _floor(minutes%60)
    if hours < 1:
        return "%d m %d s"%(minutes, seconds)
    
    days = _floor(hours/24)
    hours = _floor(hours%24)
    
    if days < 1:
        return "%d h %d m %d s"%(hours, minutes, seconds)

    return "%d days %d h %d m %d s"%(days, hours, minutes, seconds)

def tic():
    global _toc_time
    _toc_time = _time.time()

def toc():
    global _toc_time
    old_toc_time = _toc_time
    _toc_time = _time.time()
    return _toc_time - old_toc_time

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except Exception, e:
        pass
    return False

def add_iterable(iterable, initial=None):
    from operator import add
    if not is_iterable(iterable):
        error("expected an iterable")
    if initial is None:
        return reduce(add, iterable)
    return reduce(add, iterable, initial)

def camel_capitalize(name):
    return "".join(n.capitalize() for n in name.split("_"))

def tuplewrap(arg):
    "Wrap the argument to a tuple if it is not a tuple"
    if arg is None:
        return ()
    return arg if isinstance(arg, tuple) else (arg,)
    
def listwrap(arg):
    "Wrap the argument to a list if it is not a list"
    if arg is None:
        return []
    return arg if isinstance(arg, list) else [arg,]

def check_arg(arg, argtype, num=-1, itemtype=None):
    "Type check for positional arguments"
    assert(isinstance(argtype, (tuple, type)))
    if isinstance(arg, argtype):
        if itemtype is None or not isinstance(arg, (list, tuple)):
            return
        iterativetype = type(arg).__name__
        assert(isinstance(itemtype, (type, tuple)))
        if all(isinstance(item, itemtype) for item in arg):
            return
        
        assert(isinstance(num, int))
        itemtype = tuplewrap(itemtype)
        
        message = "expected a '%s' of '%s'"%(iterativetype,\
                                ", ".join(argt.__name__ for argt in itemtype))
    else:
        assert(isinstance(num, int))
        argtype = tuplewrap(argtype)
        message = "expected a '%s' (got '%s' which is a '%s')"%\
                  (", ".join(argt.__name__ for argt in argtype), \
                   str(arg), type(arg).__name__)
        
    if num != -1:
        message += " as the %s argument"%(["first", "second", "third", \
                                           "fourth", "fifth", "sixth",\
                                           "seventh", "eigth",\
                                           "ninth"][num])
    error(message)

def check_kwarg(kwarg, name, argtype, itemtype=None):
    "Type check for positional arguments"
    assert(isinstance(argtype, (tuple, type)))
    if isinstance(kwarg, argtype):
        if itemtype is None or not isinstance(kwarg, (list, tuple)):
            return
        iterativetype = type(kwarg).__name__
        assert(isinstance(itemtype, (type, tuple)))
        if all(isinstance(item, itemtype) for item in kwarg):
            return
        
        assert(isinstance(num, int))
        message = "expected a '%s' of '%s'"%(iterativetype, itemtype.__name__)
    else:
        assert(isinstance(name, str))
        argtype = tuplewrap(argtype)
        message = "expected a '%s'"%(", ".join(argt.__name__ \
                                               for argt in argtype))
        
    message += " as the '%s' argument"%name
    error(message)

def quote_join(list_of_str):
    assert(isinstance(list_of_str, (tuple, list)))
    assert(all(isinstance(item, str) for item in list_of_str))
    return ", ".join(["'%s'"%item for item in list_of_str])

__all__ = [_name for _name in globals().keys() if _name[0] != "_"]

del _name
