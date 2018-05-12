from collections import Counter, namedtuple

import numpy as np

from graph_utils import Graph
from tree_utils import Sentence


class Dictionary(Counter):
    def __init__(self, initial=("___PAD___", "___UNKNOWN___")):
        super(Dictionary, self).__init__()
        self.int_to_word = []
        self.word_to_int = {}
        self.update(initial)

    def __setitem__(self, key, value):
        if not key in self:
            self.word_to_int[key] = len(self.int_to_word)
            self.int_to_word.append(key)
        super(Dictionary, self).__setitem__(key, value)

    def lookup(self, sentences, bucket, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)

        def lookup_by_keys(obj, keys, default=1):
            result = None
            for key in keys:
                result = self.word_to_int.get(getattr(obj, key))
                if result is not None:
                    break
            if result is None:
                result = default
            return result

        result = np.zeros((len(sentences), bucket), dtype=np.int32)
        for sent_idx, sentence in enumerate(sentences):
            result[sent_idx, :len(sentence)] = [lookup_by_keys(i, keys) for i in sentence]
        return result

    def use_top_k(self, k, ensure=()):
        ret = Dictionary(initial=())
        for ensure_item in ensure:
                ret[ensure_item] = 0
        for word, count in self.most_common(k):
            ret[word] = count
        ret.int_to_word = list(ret.keys())
        ret.word_to_int = {word: idx for idx, word in enumerate(ret.int_to_word)}
        return ret

    def __getstate__(self):
        return dict(self), self.int_to_word, self.word_to_int

    def __setstate__(self, state):
        data, self.int_to_word, self.word_to_int = state
        self.update(data)

    def __reduce__(self):
        return Dictionary, ((),), self.__getstate__()


class Statistics(namedtuple("_", ["words", "postags", "labels", "characters", "supertags"])):
    @classmethod
    def from_sentences(cls, sentences):
        """:type sentences: list[Graph | Sentence]"""
        ret = cls(Dictionary(), Dictionary(), Dictionary(), Dictionary(), Dictionary())
        for sentence in sentences:
            ret.words.update(i.norm for i in sentence)
            ret.characters.update(j for i in sentence for j in i.norm)
            ret.postags.update(i.postag for i in sentence)
            ret.supertags.update(getattr(i, "supertag", None) for i in sentence)
            if isinstance(sentence, Graph):
                ret.labels.update(edge.label for node in sentence for edge in node.edges)
            else:
                assert isinstance(sentence, Sentence)
                ret.labels.update(i.relation for i in sentence)
        return ret

    def __str__(self):
        return "{} words, {} postags, {} labels, {} characters, {} supertags.".format(
            len(self.words), len(self.postags), len(self.labels), len(self.characters),
            len(self.supertags)
        )
