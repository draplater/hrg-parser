from __future__ import division

from collections import defaultdict


class CounterItem(object):
    __slots__ = ("count", "example")
    def __init__(self):
        self.count = 0
        self.example = None

    def __getstate__(self):
        return self.count, self.example

    def __setstate__(self, state):
        self.count, self.example = state

    def __iter__(self):
        return (getattr(self, i) for i in self.__slots__)

class SampleCounter(defaultdict):
    def __init__(self, item_class=CounterItem):
        super(SampleCounter, self).__init__(item_class)

    def add(self, key, value):
        item = self[key]
        item.count += 1
        item.example = value
