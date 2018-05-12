from __future__ import print_function

import sys
import attr
from collections import Counter
from typing import NamedTuple, FrozenSet

from delphin.mrs import Mrs
from delphin.mrs.eds import Eds
from hrgguru.hyper_graph import HyperGraph, GraphNode, HyperEdge
from delphin.mrs.components import nodes as mrs_nodes, links as mrs_links


@attr.s(slots=True, frozen=True, hash=False, cmp=False)
class Span(object):
    start = attr.ib(type=int)
    end = attr.ib(type=int)

    def __iter__(self):
        return iter((self.start, self.end))

    def __getitem__(self, item):
        return (self.start, self.end)[item]

    def __hash__(self):
        return hash((self.start, self.end))

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end


@attr.s(slots=True, frozen=True, hash=False, cmp=False)
class Name(object):
    span = attr.ib(type=Span)
    name = attr.ib(type=str)

    @classmethod
    def from_tuple(cls, t):
        span, name = t
        # noinspection PyArgumentList
        return cls(Span(*span), name)

    def __hash__(self):
        return hash((self.span.start, self.span.end, self.name))

    def __eq__(self, other):
        return self.span == other.span and self.name == other.name


@attr.s(slots=True, frozen=True, hash=False, cmp=False)
class Arg(object):
    from_span = attr.ib(type=Span)
    to_span = attr.ib(type=Span)
    role = attr.ib(type=str)

    @classmethod
    def from_tuple(cls, t):
        from_span, to_span, rule = t
        # noinspection PyArgumentList
        return cls(Span(*from_span), Span(*to_span), rule)

    def __hash__(self):
        return hash((self.from_span.start, self.from_span.end, self.to_span.start, self.to_span.end, self.role))

    def __eq__(self, other):
        return self.from_span == other.from_span and self.to_span == other.to_span and self.role == other.role


class ScorerResult(NamedTuple("Result", [
    ("matched_name", int),
    ("predicted_name", int),
    ("gold_name", int),
    ("matched_arg", int),
    ("predicted_arg", int),
    ("gold_arg", int)
])):

    def __add__(self, other):
        assert isinstance(other, ScorerResult)
        return self.__class__(self.matched_name + other.matched_name,
                              self.predicted_name + other.predicted_name,
                              self.gold_name + other.gold_name,
                              self.matched_arg + other.matched_arg,
                              self.predicted_arg + other.predicted_arg,
                              self.gold_arg + other.gold_arg)

    @property
    def name_precision(self):
        return self.matched_name / (self.predicted_name + sys.float_info.epsilon)

    @property
    def name_recall(self):
        return self.matched_name / (self.gold_name + sys.float_info.epsilon)

    @property
    def name_f1(self):
        return 2 * self.name_precision * self.name_recall / (self.name_precision + self.name_recall +
                                                             sys.float_info.epsilon)

    @property
    def arg_precision(self):
        return self.matched_arg / (self.predicted_arg + sys.float_info.epsilon)

    @property
    def arg_recall(self):
        return self.matched_arg / (self.gold_arg + sys.float_info.epsilon)

    @property
    def arg_f1(self):
        return 2 * self.arg_precision * self.arg_recall / (self.arg_precision + self.arg_recall +
                                                           sys.float_info.epsilon)

    @property
    def precision(self):
        return (self.matched_name + self.matched_arg) / (self.predicted_name + self.predicted_arg +
                                                         sys.float_info.epsilon)

    @property
    def recall(self):
        return (self.matched_name + self.matched_arg) / (self.gold_name + self.gold_arg + sys.float_info.epsilon)

    @property
    def f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall +
                                                   sys.float_info.epsilon)

    @classmethod
    def zero(cls):
        return cls(0, 0, 0, 0, 0, 0)

    def __str__(self):
        ret = "\tGold\tSystem\tCommon\tP\tR\tF\n"
        ret += ("#Name" + "\t{}" * 3 + "\t{:.2f}" * 3 + "\n").format(
            self.gold_name, self.predicted_name, self.matched_name,
            self.name_precision * 100, self.name_recall * 100, self.name_f1 * 100)
        ret += ("#Arg" + "\t{}" * 3 + "\t{:.2f}" * 3 + "\n").format(
            self.gold_arg, self.predicted_arg, self.matched_arg,
            self.arg_precision * 100, self.arg_recall * 100, self.arg_f1 * 100)
        ret += ("#Total" + "\t{}" * 3 + "\t{:.2f}" * 3 + "\n").format(
            self.gold_name + self.gold_arg,
            self.predicted_name + self.predicted_arg,
            self.matched_name + self.matched_arg,
            self.precision * 100, self.recall * 100, self.f1 * 100)
        ret += "\t".join("{:.2f}".format(i * 100)
                         for i in (self.name_precision, self.name_recall, self.name_f1,
                                   self.arg_precision, self.arg_recall, self.arg_f1,
                                   self.precision, self.recall, self.f1)) + "\n"
        return ret


class EDSScorer(NamedTuple("EDSScorer", [("names", FrozenSet[Name]),
                                         ("args", FrozenSet[Arg])])):
    @classmethod
    def from_eds(cls,
                 e,  # type: Eds
                 sentence_id=None
                 ):
        """:rtype: HyperGraph"""
        node_name_to_span = {}
        names = []
        args = []

        for node in e.nodes():
            span = Span(*node.lnk.data)
            node_name_to_span[node.nodeid] = span
            names.append(Name(span, str(node.pred)))

        for node in e.nodes():
            for label, target in e.edges(node.nodeid).items():
                args.append(Arg(node_name_to_span[node.nodeid],
                                node_name_to_span[target],
                                label))

        names_set = frozenset(names)
        args_set = frozenset(args)
        return cls(names_set, args_set)

    @classmethod
    def from_mrs(cls,
                 m  # type: Mrs
                 ):
        """:rtype: HyperGraph"""
        node_name_to_span = {}
        names = []
        args = []

        for node in m.eps():
            span = Span(*node.lnk.data)
            node_name_to_span[node.nodeid] = span
            names.append(Name(span, str(node.pred)))

        for start, end, rargname, post in mrs_links(m):
            if start == 0:
                continue
            args.append(Arg(node_name_to_span[start],
                            node_name_to_span[end],
                            rargname + "/" + post))

        return cls(frozenset(names), frozenset(args))

    @classmethod
    def from_hypergraph(cls,
                        hg,  # type: HyperGraph
                        log_func=print
                        ):
        node_mapping = {}  # node -> pred edge
        real_edges = []
        for edge in hg.edges:  # type: HyperEdge
            if len(edge.nodes) == 1:
                main_node = edge.nodes[0]  # type: GraphNode
                if node_mapping.get(main_node) is not None:
                    log_func("Dumplicate node name {} and {}!".format(
                        node_mapping[main_node],
                        edge.label
                    ))
                    continue
                if not edge.is_terminal:
                    log_func("non-terminal edge {} found.".format(edge.label))
                node_mapping[main_node] = edge
            elif len(edge.nodes) == 2:
                real_edges.append(edge)
            else:
                log_func("Invalid hyperedge with node count {}".format(len(edge.nodes)))

        names = []
        args = []
        for node, pred_edge in node_mapping.items():
            assert pred_edge.span is not None
            names.append(Name(Span(*pred_edge.span), pred_edge.label))

        for edge in real_edges:
            node_1, node_2 = edge.nodes
            pred_edges = [node_mapping.get(i) for i in edge.nodes]
            if any(i is None for i in pred_edges):
                log_func("No span for edge {}, nodes {}!".format(edge, pred_edges))
                continue
            args.append(Arg(Span(*pred_edges[0].span), Span(*pred_edges[1].span),
                            edge.label))

        return cls(frozenset(names), frozenset(args))

    def compare_with(self, gold, print_error=False, log_func=print):
        matched_names = self.names & gold.names
        matched_args = self.args & gold.args
        if print_error:
            log_func("Missing names: ")
            for i in gold.names - matched_names:
                log_func(i)
            log_func("Redundant names: ")
            for i in self.names - matched_names:
                log_func(i)
            log_func("Missing args: ")
            for i in gold.args - matched_args:
                log_func(i)
            log_func("Redundant args: ")
            for i in self.args - matched_args:
                log_func(i)
        return ScorerResult(len(matched_names), len(self.names), len(gold.names),
                            len(matched_args), len(self.args), len(gold.args))
