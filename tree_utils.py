from io import open

import copy
import os
import re
from collections import Counter

import numpy as np
import sh
import sys

import six

from common_utils import deprecated
from conll_reader import CoNLLUNode, CoNLLUSentence


@six.python_2_unicode_compatible
class SentenceNode(object):
    def __init__(self, id_, form, lemma, cpos, pos, feats, parent_id, relation):
        self.id = int(id_)
        self.form = form
        self.lemma = lemma
        self.norm = normalize(form)
        self.cpos = cpos
        self.pos = pos
        if self.pos == "_":
            self.pos = self.cpos
        self.feats = feats
        self.parent_id = parent_id
        self.relation = relation

    @classmethod
    def from_conllu_node(cls, conllu_node):
        return cls(conllu_node.id_, conllu_node.form, conllu_node.lemma,
                   conllu_node.cpostag.upper(),
                   conllu_node.postag.upper(), conllu_node.feats,
                   int(conllu_node.head), conllu_node.deprel)

    def copy(self):
        return copy.copy(self)

    @classmethod
    def root_node(cls):
        return cls(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot')

    @property
    def postag(self):
        return self.pos

    def __str__(self):
        return u"{}: {}-{}, head {}".format(self.id, self.form, self.pos, self.parent_id)

    def __repr__(self):
        return self.__str__()


class Sentence(list):
    NodeType = SentenceNode
    performance_pattern = re.compile(r"^(.+?)[\s|]+([\d.]+)", re.MULTILINE)

    @classmethod
    def from_conllu_sentence(cls, sent, root_last=True):
        ret = cls(cls.NodeType.from_conllu_node(i) for i in sent)
        if root_last:
            ret.append(cls.NodeType.root_node())
        else:
            ret.insert(0, cls.NodeType.root_node())
        return ret

    def copy(self):
        return self.__class__(i.copy() for i in self)

    @classmethod
    def from_file(cls, file_name, use_edge=True, root_last=False):
        with open(file_name) as f:
            return [cls.from_conllu_sentence(i, root_last)
                    for i in CoNLLUSentence.get_all_sentences(f)]

    @classmethod
    def from_words_and_postags(cls, items):
        ret = cls()
        ret.append(cls.NodeType.root_node())
        for idx, (word, postag) in enumerate(items, 1):
            ret.append(cls.NodeType(idx, word, word, postag, postag, None, None, None))
        return ret

    @staticmethod
    def evaluate_with_external_program(gold_file, output_file):
        current_path = os.path.dirname(__file__)
        eval_script = os.path.join(current_path, "utils/evaluation_script/conll17_ud_eval.py")
        weight_file = os.path.join(current_path, "utils/evaluation_script/weights.clas")
        eval_process = sh.python(eval_script, "-v", "-w", weight_file,
                                 gold_file, output_file, _out=output_file + '.txt')
        eval_process.wait()
        sh.cat(output_file + '.txt', _out=sys.stdout)

    @classmethod
    def extract_performance(cls, perf_file_name):
        with open(perf_file_name) as f:
            content = f.read()

        def generate_items():
            for k, v in cls.performance_pattern.findall(content):
                yield k, float(v)

        result = dict(generate_items())
        epoch = re.findall(r"epoch_(\d+)[_.]", perf_file_name)[0]
        result["epoch"] = int(epoch)
        return result


    def to_matrix(self):
        ret = np.zeros((len(self), len(self)), dtype=np.bool)
        for dep, head in enumerate((i.parent_id for i in self[1:]), 1):
            ret[head, dep] = 1
        return ret


@deprecated
def vocab(sentences):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()

    for sentence in sentences:
        wordsCount.update([node.norm for node in sentence if isinstance(node, SentenceNode)])
        posCount.update([node.pos for node in sentence if isinstance(node, SentenceNode)])
        relCount.update([node.relation for node in sentence if isinstance(node, SentenceNode)])

    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())},
            posCount.keys(), relCount.keys())


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()