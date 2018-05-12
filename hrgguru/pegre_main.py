
import re
from functools import wraps
from collections import Sequence, namedtuple

import six

__all__ = [
    'Ignore',
    'literal',
    'regex',
    'nonterminal',
    'and_next',
    'not_next',
    'sequence',
    'choice',
    'optional',
    'zero_or_more',
    'one_or_more',
    'Peg',
]

PegreFail = namedtuple('PegreFail', ('string', 'reason'))
Ignore = object()  # just a singleton for identity checking

def valuemap(f):
    """
    Decorator to help PEG functions handle value conversions.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'value' in kwargs:
            val = kwargs['value']
            del kwargs['value']
            _f = f(*args, **kwargs)
            def valued_f(*args, **kwargs):
                result = _f(*args, **kwargs)
                if not isinstance(result, PegreFail):
                    s, obj = result
                    if callable(val):
                        return (s, val(obj))
                    else:
                        return (s, val)
                else:
                    return result
            return valued_f
        else:
            return f(*args, **kwargs)
    return wrapper

@valuemap
def literal(x):
    """
    Create a PEG function to consume a literal.
    """
    xlen = len(x)
    msg = 'Expected: "{}"'.format(x)
    def match_literal(s, grm):
        if s[:xlen] == x:
            return (s[xlen:], x)
        return PegreFail(s, msg)
    return match_literal

@valuemap
def regex(r):
    """
    Create a PEG function to match a regular expression.
    """
    if isinstance(r, six.string_types):
        p = re.compile(r)
    else:
        p = r
    msg = 'Expected to match: {}'.format(p.pattern)
    def match_regex(s, grm):
        m = p.match(s)
        if m is not None:
            return (s[m.end():], m.group())
        return PegreFail(s, msg)
    return match_regex

@valuemap
def nonterminal(n):
    """
    Create a PEG function to match a nonterminal.
    """
    def match_nonterminal(s, grm):
        expr = grm[n]
        return expr(s, grm)
    return match_nonterminal

@valuemap
def and_next(e):
    """
    Create a PEG function for positive lookahead.
    """
    def match_and_next(s, grm):
        if e(s, grm):
            return (s, Ignore)
        return PegreFail(s, repr(e))
    return match_and_next

@valuemap
def not_next(e):
    """
    Create a PEG function for negative lookahead.
    """
    def match_not_next(s, grm):
        if not e(s, grm):
            return (s, Ignore)
        return PegreFail(s, '! ' + repr(e))
    return match_not_next

@valuemap
def sequence(*es):
    """
    Create a PEG function to match a sequence.
    """
    def match_sequence(s, grm):
        data = []
        for e in es:
            result = e(s, grm)
            if isinstance(result, PegreFail):
                return result
            s, obj = result
            if obj is not Ignore:
                data.append(obj)
        return (s, data)
    return match_sequence

@valuemap
def choice(*es):
    """
    Create a PEG function to match an ordered choice.
    """
    msg = 'Expected one of: {}'.format(', '.join(map(repr, es)))
    def match_choice(s, grm):
        for e in es:
            result = e(s, grm)
            if not isinstance(result, PegreFail):
                return result
        return PegreFail(s, msg)
    return match_choice

@valuemap
def optional(e):
    """
    Create a PEG function to optionally match an expression.
    """
    def match_optional(s, grm):
        result = e(s, grm)
        if not isinstance(result, PegreFail):
            return result
        return (s, Ignore)
    return match_optional

@valuemap
def zero_or_more(e, delimiter=None):
    """
    Create a PEG function to match zero or more expressions.

    Args:
        e: the expression to match
        delimiter: an optional expression to match between the
            primary *e* matches.
    """
    def match_zero_or_more(s, grm):
        result = e(s, grm)
        if not isinstance(result, PegreFail):
            data = []
            while not isinstance(result, PegreFail):
                s, obj = result
                if obj is not Ignore:
                    data.append(obj)
                if delimiter is not None:
                    result = delimiter(s, grm)
                    if isinstance(result, PegreFail):
                        break
                    s, obj = result
                    if obj is not Ignore:
                        data.append(obj)
                result = e(s, grm)
            return (s, data)
        return (s, Ignore)
    return match_zero_or_more

@valuemap
def one_or_more(e, delimiter=None):
    """
    Create a PEG function to match one or more expressions.

    Args:
        e: the expression to match
        delimiter: an optional expression to match between the
            primary *e* matches.
    """
    msg = 'Expected one or more of: {}'.format(repr(e))
    def match_one_or_more(s, grm):
        result = e(s, grm)
        if not isinstance(result, PegreFail):
            data = []
            while not isinstance(result, PegreFail):
                s, obj = result
                if obj is not Ignore:
                    data.append(obj)
                if delimiter is not None:
                    result = delimiter(s, grm)
                    if isinstance(result, PegreFail):
                        break
                    s, obj = result
                    if obj is not Ignore:
                        data.append(obj)
                result = e(s, grm)
            return (s, data)
        return PegreFail(s, msg)
    return match_one_or_more

@valuemap
def bounded(pre, expr, post):
    return sequence(pre, expr, post, value=lambda x: x[1])

@valuemap
def delimited(expr, delim):
    return zero_or_more(expr, delimiter=delim, value=lambda x: x[::2])


class Peg(object):
    """
    A class to assist in parsing using a grammar of PEG functions.
    """
    def __init__(self, grammar, start='start'):
        self.start = start
        self.grammar = grammar

    def parse(self, s):
        result = self.grammar[self.start](s, self.grammar)
        if isinstance(result, PegreFail):
            raise ValueError(repr(result))
        else:
            return result[1]
