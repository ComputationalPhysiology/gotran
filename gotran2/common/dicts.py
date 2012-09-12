__author__ = "Johan Hake (hake.dev@gmail.com)"
__copyright__ = "Copyright (C) 2010 " + __author__
__date__ = "2008-09-16 -- 2011-09-27"
__license__  = "GNU LGPL Version 3.0 or later"

__all__ = ["adict", "odict"]

class adict(dict):
    """A dictionary with attribute-style access. It maps attribute access to
    the real dictionary.  """
    def __init__(self, **kwargs):
        dict.__init__(self, **kwargs)
        
    def __getstate__(self):
        return self.__dict__.items()
    
    def __setstate__(self, items):
        for key, val in items:
            self.__dict__[key] = val
    
    def __repr__(self):
        return dict.__repr__(self)
    
    def __setitem__(self, key, value):
        return super(adict, self).__setitem__(key, value)
    
    def __getitem__(self, name):
        return super(adict, self).__getitem__(name)
    
    def __delitem__(self, name):
        return super(adict, self).__delitem__(name)
    
    __getattr__ = __getitem__
    __setattr__ = __setitem__
    
    def copy(self):
        ch = adict(self)
        return ch

class odict(dict):
    "A simple ordered dict"
    def __init__(self, iterable=None):
        self._keys = []
        dict.__init__(self)
        if iterable is not None:
            for key, value in iterable:
                self[key] = value

    def __setitem__(self, key, value):
        if not key in self._keys:
            self._keys.append(key)
        dict.__setitem__(self, key, value)

    def keys(self):
        return self._keys

    def __iter__(self):
        return self.iterkeys()

    def iterkeys(self):
        for key in self._keys:
            yield key

    def iteritems(self):
        for key in self._keys:
            yield key, self[key]

    def itervalues(self):
        for key in self._keys:
            yield self[key]

    def values(self):
        return [self[key] for key in self._keys]

    def items(self):
        return [(key, self[key]) for key in self._keys]

    def pop(self, *args, **kwargs):
        return NotImplemented
    
    def popitem(self, *args, **kwargs):
        return NotImplemented

    def setdefault(self, *args, **kwargs):
        return NotImplemented

    def update(self, *args, **kwargs):
        return NotImplemented

    def clear(self):
        self._keys = []
        dict.clear(self)

