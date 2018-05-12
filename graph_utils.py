from __future__ import unicode_literals

import tempfile
from io import open
from typing import Iterable, List

import os
import attr
from collections import Counter, namedtuple
import re
import numpy as np

from common_utils import deprecated
from conll_reader import OldSDPNode, OldSDPSentence, SDPSentence, SDPNode

Edge = namedtuple("Edge", ["source", "label", "target"])


@attr.s(slots=True)
class GraphNode(object):
    id = attr.ib(type=int)
    form = attr.ib(type=str)
    lemma = attr.ib(type=str)
    postag = attr.ib(type=str)
    top = attr.ib(type=bool)
    pred = attr.ib(type=bool)
    edges = attr.ib(type=list)
    norm = attr.ib(type=str, init=False)

    def __attrs_post_init__(self):
        self.norm = normalize(self.form)

    @classmethod
    def root_node(cls):
        # noinspection PyArgumentList
        return cls(0, "ROOT", "ROOT", "ROOT", None, None, [])

    @classmethod
    def from_sdp_node(cls, node):
        # noinspection PyArgumentList
        return cls(int(node.id_), node.form, node.lemma, node.postag,
                   node.top == "+", node.pred == "+", [])

    @classmethod
    def from_word_and_postag(cls, idx, word, postag):
        # noinspection PyArgumentList
        return cls(idx, word, word, postag, False, False, [])

    def to_sdp_node(self, arg):
        return OldSDPNode(
            self.id, self.form, self.lemma, self.postag,
            "+" if self.top else "-", "+" if self.pred else "-", arg)

    def __str__(self):
        return "{}: {}({})".format(self.id, self.form, self.postag)

    def __repr__(self):
        return self.__str__()


@attr.s(slots=True)
class GraphNode2015(GraphNode):
    sense = attr.ib(type=str)

    @property
    def supertag(self):
        return self.sense

    @classmethod
    def root_node(cls):
        # noinspection PyArgumentList
        return cls(0, "ROOT", "ROOT", "ROOT", None, None, [], "ROOT")

    @classmethod
    def from_sdp_node(cls, node):
        # noinspection PyArgumentList
        return cls(int(node.id_), node.form, node.lemma, node.postag,
                   node.top == "+", node.pred == "+", [], node.sense)

    @classmethod
    def from_word_and_postag(cls, idx, word, postag):
        # noinspection PyArgumentList
        return cls(idx, word, word, postag, False, False, [], postag)

    def to_sdp_node(self, arg):
        return SDPNode(
            self.id, self.form, self.lemma, self.postag,
            "+" if self.top else "-", "+" if self.pred else "-",
            self.sense, arg)

    def __str__(self):
        return "{}: {}({})".format(self.id, self.form, self.postag)

    def __repr__(self):
        return self.__str__()


class Graph(list):  # type: List[GraphNode]
    performance_pattern = re.compile(r"^(.+?): ([\d.]+|NaN)", re.MULTILINE)
    GraphNodeType = GraphNode
    SDPSentenceClass = OldSDPSentence
    __slots__ = ("comment", )

    def __init__(self, seq=()):
        super(Graph, self).__init__(seq)
        self.comment = "no comment"

    @classmethod
    def from_sdp(cls, sdp_sentence, use_edge=True):
        node_list = cls()
        node_list.append(cls.GraphNodeType.root_node())

        if use_edge:
            pred_list = []  # type: List[cls.GraphNodeType]
            for i in sdp_sentence:
                node = cls.GraphNodeType.from_sdp_node(i)
                node_list.append(node)
                if node.top:
                    node_list[0].edges.append(Edge(0, "ROOT", node.id))
                if node.pred:
                    pred_list.append(node)

            for token in sdp_sentence:
                assert len(pred_list) == len(token.arg), \
                    "Invalid sentence {} on {}||{}||{}".format(
                        [token.form for token in sdp_sentence], token.form, pred_list, token.arg)
                for pred_node, label in zip(pred_list, token.arg):
                    if label != '_':
                        pred_node.edges.append(Edge(pred_node.id, label, int(token.id_)))
                        assert pred_node.id <= len(sdp_sentence), [token.form for token in sdp_sentence]
                        assert int(token.id_) <= len(sdp_sentence), [token.form for token in sdp_sentence]
        else:
            for i in sdp_sentence:
                node = cls.GraphNodeType.from_sdp_node(i)
                node_list.append(node)

        node_list.comment = sdp_sentence.comment
        return node_list

    def generate_edges(self):
        for source in self:
            for i in source.edges:
                yield i

    def to_matrix(self):
        """
        :rtype np.ndarray
        """
        ret = np.zeros((len(self), len(self)), dtype=bool)
        for source, label, target in self.generate_edges():
            ret[source][target] = 1
        return ret

    def is_1ec2p(self):
        pass

    def replaced_edges(self,
                       edges  # type: Iterable[Edge]
                       ):
        new_obj = self.__class__()
        new_obj.comment = self.comment
        new_obj.append(self.GraphNodeType.root_node())
        for node in self[1:]:
            new_obj.append(attr.evolve(node, top=False, pred=False, edges=[]))
        for edge in edges:
            if edge.source == 0:
                new_obj[edge.target].top = True
            elif edge.source > 0:
                new_obj[edge.source].pred = True
            else:
                raise RuntimeError()
            new_obj[edge.source].edges.append(edge)
        return new_obj

    def get_args_list(self):
        args = [[] for _ in range(len(self) - 1)]

        pred_count = 0
        for node in self:
            if node.pred:
                for edge in node.edges:
                    args[edge.target - 1].append(edge.label)
                pred_count += 1
                for token_args in args:
                    if len(token_args) != pred_count:
                        token_args.append("_")
        return args

    def to_sdp(self):
        args = self.get_args_list()
        ret = self.SDPSentenceClass()
        for idx, node in enumerate(self[1:]):
            sdp_node = node.to_sdp_node(args[idx])
            ret.append(sdp_node)
        ret.comment = self.comment
        return ret

    @classmethod
    def from_file(cls, file_name, use_edge=True):
        with open(file_name, "r") as f:
            graphs = [cls.from_sdp(i, use_edge)
                      for i in cls.SDPSentenceClass.get_all_sentences(f)]
        return graphs

    @classmethod
    def from_words_and_postags(cls, items):
        ret = cls()
        ret.append(cls.GraphNodeType.root_node())
        for idx, (word, postag) in enumerate(items, 1):
            ret.append(cls.GraphNodeType.from_word_and_postag(idx, word, postag))
        return ret

    @classmethod
    def preprocess_gold_file(cls, gold_file, limit=None):
        if limit is not None:
            gold_tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
            data = cls.from_file(gold_file)
            for sent in data[:limit]:
                sent.comment = "no comment"
                gold_tmp.write(sent.to_string())
            gold_tmp.close()
            return gold_tmp
        else:
            return gold_file

    @classmethod
    def evaluate_with_external_program(cls, gold_file, output_file, limit=None):
        gold_tmp = cls.preprocess_gold_file(gold_file, limit)
        current_path = os.path.dirname(__file__)
        os.system('sh {}/utils/sdptool/run.sh Scorer {} {} 2> {}.txt'.format(
            current_path, gold_tmp, output_file, output_file))
        os.system('cat {}.txt'.format(output_file))
        if gold_tmp != gold_file:
            os.remove(gold_tmp)

    @classmethod
    def extract_performance(cls, perf_file_name):
        with open(perf_file_name) as f:
            content = f.read()

        def strip_nan(x):
            if x == x:
                return x
            return -1

        def generate_items():
            for k, v in cls.performance_pattern.findall(content):
                yield k, strip_nan(float(v))

        result = dict(generate_items())
        epoch = re.findall(r"epoch_(\d+)[_.]", perf_file_name)[0]
        result["epoch"] = int(epoch)
        return result

    def to_string(self):
        return self.to_sdp().to_string()


class Graph2015(Graph):
    GraphNodeType = GraphNode2015
    SDPSentenceClass = SDPSentence
    file_header = "#SDP 2015"
    __slots__ = ()

    @classmethod
    def evaluate_with_external_program(cls, gold_file, output_file, limit=None):
        gold_tmp = cls.preprocess_gold_file(gold_file, limit)
        current_path = os.path.dirname(__file__)
        os.system('java -cp {}/utils/sdp2015.jar '
                  'se.liu.ida.nlp.sdp.toolkit.tools.Scorer '
                  '{} {} 2> {}.txt'.format(
            current_path, gold_tmp, output_file, output_file))
        os.system('cat {}.txt'.format(output_file))
        if gold_tmp != gold_file:
            os.remove(gold_tmp)


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()

