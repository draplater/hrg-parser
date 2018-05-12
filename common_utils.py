from io import open
import contextlib
import functools
import warnings
from argparse import ArgumentParser
from optparse import OptionParser

import os
import time

import sys

from itertools import islice


def set_proc_name(newname):
    from ctypes import cdll, byref, create_string_buffer
    libc = cdll.LoadLibrary('libc.so.6')
    buff = create_string_buffer(len(newname)+1)
    buff.value = newname.encode("utf-8")
    libc.prctl(15, byref(buff), 0, 0, 0)


def ensure_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as err:
        if err.errno!=17:
            raise


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning) #turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning) #reset filter
        return func(*args, **kwargs)

    return new_func


@deprecated
def parse_dict(parser, dic, prefix=()):
    from training_scheduler import dict_to_commandline
    return parser.parse_args(dict_to_commandline(dic, prefix))


def under_construction(func):
    """This is a decorator which can be used to mark functions
    as under construction. It will result in a warning being emmitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn("Call to under construction function {}.".format(func.__name__), category=UserWarning, stacklevel=2)
        return func(*args, **kwargs)

    return new_func


class Timer(object):
    def __init__(self):
        self.time = time.time()

    def tick(self):
        oldtime = self.time
        self.time = time.time()
        return self.time - oldtime


@contextlib.contextmanager
def smart_open(filename, mode="r", *args, **kwargs):
    if filename != '-':
        fh = open(filename, mode, *args, **kwargs)
    else:
        if mode.startswith("r"):
            fh = sys.stdin
        elif mode.startswith("w") or mode.startswith("a"):
            fh  = sys.stdout
        else:
            raise ValueError("invalid mode " + mode)

    try:
        yield fh
    finally:
        if fh is not sys.stdout and fh is not sys.stdin:
            fh.close()


def split_to_batches(iterable, batch_size):
    iterator = iter(iterable)
    sent_id = 0
    batch_id = 0

    while True:
        piece = list(islice(iterator, batch_size))
        if not piece:
            break
        yield sent_id, batch_id, piece
        sent_id += len(piece)
        batch_id += 1


class AttrDict(dict):
    """A dict whose items can also be accessed as member variables.

    >>> d = AttrDict(a=1, b=2)
    >>> d['c'] = 3
    >>> print d.a, d.b, d.c
    1 2 3
    >>> d.b = 10
    >>> print d['b']
    10

    # but be careful, it's easy to hide methods
    >>> print d.get('c')
    3
    >>> d['get'] = 4
    >>> print d.get('a')
    Traceback (most recent call last):
    TypeError: 'int' object is not callable
    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    @property
    def __dict__(self):
        return self


class IdentityDict(object):
    """ A dict like IdentityHashMap in java"""
    def __init__(self, seq=None):
        self.dict = dict(seq=((id(key), value) for key, value in seq))

    def __setitem__(self, key, value):
        self.dict[id(key)] = value

    def __getitem__(self, item):
        return self.dict[id(item)]

    def get(self, key, default=None):
        return self.dict.get(id(key), default)

    def __str__(self):
        return str(self.dict)

    def __repr__(self):
        return repr(self.dict)

    def __getstate__(self):
        raise NotImplementedError("Cannot pickle this.")

