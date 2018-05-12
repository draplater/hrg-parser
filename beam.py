import heapq
from operator import attrgetter


class Beam(object):
    def __init__(self, maxsize, key=attrgetter("score")):
        self.key = key
        self.maxsize = maxsize
        self.beam = []

    def push(self, x):
        key = self.key(x)
        if len(self.beam) < self.maxsize:
            heapq.heappush(self.beam, (key, x))
        else:
            worst_score, worst_state = self.beam[0]
            if key > worst_score:
                heapq.heapreplace(self.beam, (key, x))

    def __len__(self):
        return len(self.beam)

    def full(self):
        return len(self.beam) == self.maxsize

    def empty(self):
        return not len(self.beam)

    def clear(self):
        self.beam.clear()

    def __iter__(self):
        return iter(i[1] for i in self.beam)

    def __getitem__(self, item):
        return self.beam[item][1]

    def best_item(self):
        return sorted(self.beam, reverse=True)[0][1]